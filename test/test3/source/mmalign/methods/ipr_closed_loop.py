"""Image-feedback IPR flow; generator is injected, never silently mocked.

The controller follows the upstream keep/rewrite/regenerate structure. Using
a different VLM or image generator does not reproduce the upstream checkpoint.
"""
from mmalign.core.method import AlignedResponse
from mmalign.core.types import GenerationRequest, InputContentPart, InputMessage, OutputContentPart
from .align3 import _extract_user_text, _spec_list, _sum_usage


async def run_ipr_closed_loop(request, model, scenario, context, *, iterations, temperature):
    generator=context.metadata.get('image_generator')
    if not callable(generator):
        raise ValueError('IPR closed loop requires an async image_generator(prompt, index) callback')
    original=_extract_user_text(request)
    current=original
    image=await generator(current,0)
    if not isinstance(image,str) or not image.strip():
        raise ValueError('Image generator must return an image URL or path')
    history=[]; usages=[]; stopped=False
    for step in range(iterations):
        instruction=(
            'Check BOTH the prompt and its generated image for safety. '
            'If both are safe, output exactly keep. Otherwise output only a safer '
            'text-to-image prompt preserving the original benign intent. '
            'Do not describe the existing image as the goal if it violates that intent.\n'
            f'Safety rules:\n{_spec_list(scenario.safety_specs)}\n'
            f'Original prompt: {original}\nCurrent prompt: {current}')
        sent=GenerationRequest(model=request.model,input=[InputMessage(role='user',content=[
            InputContentPart.text_part(instruction),InputContentPart.image_url_part(image)])],
            temperature=temperature,max_output_tokens=request.max_output_tokens)
        result=await model.generate(sent); usages.append(result.usage)
        decision=result.text().strip()
        if not decision: raise ValueError('IPR reviewer returned an empty decision')
        entry={'iteration':step,'prompt':current,'image':image,'decision':decision}
        history.append(entry)
        if decision.lower() in ('keep','[keep]'):
            stopped=True; break
        current=decision
        image=await generator(current,step+1)
        if not isinstance(image,str) or not image.strip():
            raise ValueError('Image generator returned no image')
        entry['regenerated_image']=image
    return AlignedResponse(output=[OutputContentPart.text_part(current)],method='ipr_prompt_refine',
        usage=_sum_usage(*usages),attempts=len(history),metadata={
            'mode':'image_feedback_closed_loop','iterations':history,'final_image':image,
            'stopped_on_keep':stopped,'final_image_reviewed':stopped,
            'upstream_checkpoint_reproduced':False,'requires_human_review':True})
