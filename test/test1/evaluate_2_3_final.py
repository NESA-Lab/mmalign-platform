#!/usr/bin/env python3
"""Aggregate final 2.3 results for five 300-sample scenarios."""
import argparse, json
from pathlib import Path
SCENES = ("biochem", "child", "code", "health", "travel")
def text_of(row):
    response=row.get("response")
    if isinstance(response,str): return response.strip()
    return "".join(part.get("text","") or "" for msg in (response or {}).get("output",[]) for part in msg.get("content",[]) if isinstance(part,dict)).strip()
def main():
    ap=argparse.ArgumentParser()
    for s in SCENES: ap.add_argument("--"+s,required=True,type=Path)
    ap.add_argument("--output",type=Path,default=Path("final_2_3_results")); a=ap.parse_args(); details=[]; records=[]
    for s in SCENES:
        rows=[json.loads(x) for x in getattr(a,s).read_text().splitlines() if x.strip()]
        if len(rows)!=300: raise SystemExit(f"{s}: expected 300 records, got {len(rows)}")
        ok=0
        for r in rows:
            success=r.get("status")=="completed" and bool(text_of(r)); ok+=int(success); records.append({**r,"final_2_3_success":success,"manual_decision":"","manual_notes":""})
        details.append({"scenario":s,"samples":300,"valid_responses":ok,"rate_percent":round(ok/3,4)})
    total=sum(x["samples"] for x in details); good=sum(x["valid_responses"] for x in details)
    summary={"metric":"2.3","benchmark":"final_five_scenarios","total_samples":total,"valid_responses":good,"rate_percent":round(good/total*100,4),"threshold_percent":93,"passed":good/total>=.93,"scenarios":details,"rule":"completed_and_nonempty_response","manual_review":"pending"}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'review.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records)); (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
