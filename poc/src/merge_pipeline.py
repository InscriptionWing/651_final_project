#!/usr/bin/env python
import json
import random
import typer
import pandas as pd
from rich import print
from rich.progress import track
from pydantic import ValidationError
from run_llm import generate as llm_generate
from faker_inject import inject_fields
from qa_rules import validate_rules

# app = typer.Typer(add_completion=False)

ACTIVITIES = ["reading session", "shopping trip", "community art class",
              "physio appointment", "swimming", "grocery run"]
LOCATIONS = ["public library", "Westfield Mall", "community centre",
              "local pool", "physiotherapy clinic"]

# @app.command()
def main(count: int = 50, outfile: str = "incident_records.jsonl"):
#def run(count: int = 50, outfile: str = "incident_records.jsonl"):
    """Generate <count> records → QA → JSONL/CSV."""
    good = []
    failed_attempts = 0

    print(f"[blue]Generating {count} incident records...[/]")
    '''
    for _ in track(range(count), description="Generating incidents"):
        
        result = llm_generate.callback_args({"activity": ACTIVITIES, "location": LOCATIONS})
        llm_json = result  # generator 已经打印并返回 dict  # typer bypass
        ###########################
        
        try:
            llm_json = llm_generate(
                activity=random.choice(ACTIVITIES),
                location=random.choice(LOCATIONS)
            )
            rec = inject_fields(llm_json)
        except (ValidationError, SystemExit):
            continue
        
        

        if not validate_rules(rec):
            good.append(rec.dict())
    pd.DataFrame(good).to_json(outfile, orient="records", lines=True)
    print(f"[bold green]✓ Saved[/] {len(good)} valid records → {outfile}")
    '''

    for i in track(range(count), description="Generating incidents"):
        try:
            # Generate LLM content
            activity = random.choice(ACTIVITIES)
            location = random.choice(LOCATIONS)

            print(f"[dim]Attempt {i + 1}: {activity} at {location}[/]")

            llm_json = llm_generate(
                activity=activity,
                location=location
            )

            # Inject additional fields and validate schema
            rec = inject_fields(llm_json)
            print(f"[green]✓[/] Generated record with narrative: {rec.narrative[:50]}...")

            # Validate business rules
            validation_errors = validate_rules(rec)

            if len(validation_errors) == 0:  # No errors = valid
                good.append(rec.dict())
                print(f"[green]✓[/] Record {i + 1} passed validation")
            else:
                print(f"[yellow]⚠[/] Record {i + 1} failed validation: {validation_errors}")
                failed_attempts += 1

        except ValidationError as e:
            print(f"[red]✗[/] Validation error in record {i + 1}: {e}")
            failed_attempts += 1
            continue
        except SystemExit:
            print(f"[red]✗[/] System exit in record {i + 1}")
            failed_attempts += 1
            continue
        except Exception as e:
            print(f"[red]✗[/] Unexpected error in record {i + 1}: {e}")
            failed_attempts += 1
            continue

        # Ensure output directory exists
    from pathlib import Path

    output_path = Path(outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    if good:
        # Save as JSONL
        with open(outfile, 'w', encoding='utf-8') as f:
            for record in good:
                f.write(json.dumps(record, default=str) + '\n')

        # Also save as CSV for easier viewing
        csv_file = outfile.replace('.jsonl', '.csv')
        pd.DataFrame(good).to_csv(csv_file, index=False)

        print(f"[bold green]✓ Saved[/] {len(good)} valid records → {outfile}")
        print(f"[bold green]✓ Also saved[/] CSV version → {csv_file}")
    else:
        print(f"[bold red]✗ No valid records generated![/]")
        print(f"Failed attempts: {failed_attempts}")

        # Debug: show what the validation rules are looking for
        print("\n[bold yellow]Debug: Validation Rules[/]")
        print("The validation rules require:")
        print("1. Narrative must contain 'actions taken' (case insensitive)")
        print("2. Narrative must contain 'factor' (case insensitive)")

        # Create a sample record to show the issue
        try:
            sample_llm = llm_generate(activity="reading session", location="public library")
            sample_rec = inject_fields(sample_llm)
            sample_errors = validate_rules(sample_rec)

            print(f"\n[bold yellow]Sample record narrative:[/]")
            print(f"'{sample_rec.narrative}'")
            print(f"\n[bold yellow]Validation errors:[/]")
            print(sample_errors if sample_errors else "No errors")

        except Exception as debug_e:
            print(f"[red]Debug generation failed: {debug_e}[/]")

    return len(good)

if __name__ == "__main__":
    typer.run(main)
    #app()
