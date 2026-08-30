"""Build website results JSON from archived Oak cards and a final-results CSV."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from oak_nfl.results import grade_predictions, summarize_results


def clean(v):
    if pd.isna(v): return None
    return v.item() if hasattr(v, "item") else v


def main():
    p=argparse.ArgumentParser(); p.add_argument("predictions", type=Path); p.add_argument("finals", type=Path); p.add_argument("--output", type=Path, default=Path("site/data/results.json")); a=p.parse_args()
    pred=pd.read_csv(a.predictions); finals=pd.read_csv(a.finals); graded=grade_predictions(pred, finals); weekly=summarize_results(graded)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"weekly":[{k:clean(v) for k,v in r.items()} for r in weekly.to_dict("records")],"games":[{k:clean(v) for k,v in r.items()} for r in graded.to_dict("records")]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(f"Saved {a.output} with {len(graded)} graded games")
if __name__=="__main__": main()
