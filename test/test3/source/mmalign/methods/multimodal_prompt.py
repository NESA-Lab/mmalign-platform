"""Keep original visual inputs when an alignment method rewrites its text."""
from mmalign.core.types import GenerationRequest, InputContentPart, InputMessage


def user_with_images(request: GenerationRequest, text: str) -> InputMessage:
    images = [part for message in request.input_messages()
              if message.role == 'user' and not isinstance(message.content, str)
              for part in message.content if part.type == 'input_image']
    return InputMessage(role='user', content=[InputContentPart.text_part(text), *images])
