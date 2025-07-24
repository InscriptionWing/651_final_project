#!/usr/bin/env python
import json
import random
import typer
import pandas as pd
from rich.progress import track

from run_llm import generate as llm_generate
from faker_inject import inject_fields
from qa_rules import validate_rules

app = typer.Typer(add_completion=False)

ACTIVITIES = ["reading session", "shopping trip", "community art class",
              "physio appointment", "swimming", "grocery run"]
LOCATIONS = ["public library", "Westfield Mall", "community centre",
              "local pool", "physiotherapy clinic"]

@app.command()
def run(count: int = 50, outfile: str = "incident_records.jsonl"):
    """Generate <count> records → QA → JSONL/CSV."""
    good = []
    for _ in track(range(count), description="Generating incidents"):
        result = llm_generate.callback_args({"activity": ACTIVITIES, "location": LOCATIONS})
        llm_json = result  # generator 已经打印并返回 dict  # typer bypass

        rec = inject_fields(llm_json)
        if not (issues := validate_rules(rec)):
            good.append(rec.dict())
    pd.DataFrame(good).to_json(outfile, orient="records", lines=True)
    print(f"[bold green]✓ Saved[/] {len(good)} valid records → {outfile}")

if __name__ == "__main__":
    app()
