#!/usr/bin/env python
"""
Fast LLM pipeline optimized for RTX 4070 laptop.
Target: 100 records in <30 minutes using real LLM generation.
"""
import json
import random
import typer
import pandas as pd
from rich import print
from rich.progress import track
from pydantic import ValidationError
from faker_inject import inject_fields
from qa_rules import validate_rules
import os
from pathlib import Path
import time
import concurrent.futures
from typing import List, Dict, Tuple

# Configure for LLM mode with optimizations
os.environ["USE_TEMPLATES_ONLY"] = "false"
os.environ["LLM_MODEL_CONFIG"] = os.getenv("LLM_MODEL_CONFIG", "mistral-7b-fast")

# Import optimized LLM functions
from run_llm import get_generator, OptimizedLLMGenerator

ACTIVITIES = ["reading session", "shopping trip", "community art class",
              "physio appointment", "swimming", "grocery run"]
LOCATIONS = ["public library", "Westfield Mall", "community centre",
              "local pool", "physiotherapy clinic"]


def generate_batch(generator: OptimizedLLMGenerator,
                  activity_location_pairs: List[Tuple[str, str]],
                  batch_size: int = 4) -> List[dict]:
    """Generate a batch of records using the LLM with proper batching"""
    results = []
    
    # Process pairs in actual batches
    for i in range(0, len(activity_location_pairs), batch_size):
        batch_pairs = activity_location_pairs[i:i + batch_size]
        
        try:
            # Use the first pair's activity/location for the batch, but generate multiple records
            activity, location = batch_pairs[0]
            
            # Generate batch_size records at once
            batch_results = generator.generate(activity, location, batch_size=batch_size)
            
            if batch_results:
                # Take up to batch_size results
                results.extend(batch_results[:len(batch_pairs)])
                
        except Exception as e:
            print(f"[yellow]Batch failed for {len(batch_pairs)} records: {e}[/]")
            # Try individual generation as fallback
            for activity, location in batch_pairs:
                try:
                    single_result = generator.generate(activity, location, batch_size=1)
                    if single_result:
                        results.extend(single_result)
                except Exception as single_e:
                    print(f"[red]Failed to generate for {activity} at {location}: {single_e}[/]")

    return results


def process_record(llm_json: dict) -> Tuple[dict, List[str]]:
    """Process a single LLM output into a validated record"""
    try:
        rec = inject_fields(llm_json)
        validation_errors = validate_rules(rec)

        if len(validation_errors) == 0:
            return rec.model_dump(), []
        else:
            return None, validation_errors
    except Exception as e:
        return None, [str(e)]


def main(count: int = 100,
         outfile: str = "incident_records.jsonl",
         model_config: str = None,
         batch_size: int = 4,
         parallel_workers: int = 1):
    """
    Generate records using optimized LLM pipeline.

    Args:
        count: Number of records to generate
        outfile: Output file path
        model_config: Model configuration (tinyllama/phi-2/mistral-7b-fast/mistral-7b)
        batch_size: Number of prompts to process together
        parallel_workers: Number of parallel generation workers
    """

    # Use model config from env or parameter
    if model_config is None:
        model_config = os.getenv("LLM_MODEL_CONFIG", "mistral-7b-fast")
    
    # Check VRAM and recommend smaller models if needed
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory < 8:  # Less than 8GB VRAM
                print(f"[yellow]⚠[/] Detected {gpu_memory:.1f}GB VRAM. Consider using smaller models:")
                print(f"  • 'tinyllama' (1.1B) - Fastest, lowest VRAM")
                print(f"  • 'phi-2' (2.7B) - Balanced speed/quality")
                print(f"  • 'mistral-7b-fast' (8-bit) - Current choice")
                if model_config == "mistral-7b":
                    print(f"[red]⚠[/] 'mistral-7b' may cause OOM with {gpu_memory:.1f}GB VRAM")
    except:
        pass  # Skip if torch not available

    # Optimize batch size based on model
    if model_config == "mistral-7b" and batch_size > 2:
        print(f"[yellow]Warning: Large batch size ({batch_size}) may cause OOM with full Mistral-7B. Reducing to 2.[/]")
        batch_size = 2
    elif model_config == "mistral-7b-fast" and batch_size < 4:
        print(f"[blue]Optimizing: Using batch size 4 for mistral-7b-fast (8-bit quantized)[/]")
        batch_size = 4

    print(f"[bold blue]Fast LLM Generation Pipeline[/]")
    print(f"[blue]Configuration:[/]")
    print(f"  • Model: {model_config}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Target: {count} records")
    print(f"  • Output: {outfile}")

    # Initialize generator
    print(f"\n[blue]Initializing LLM generator...[/]")
    generator = get_generator(model_config)

    # Start timing
    start_time = time.time()

    # Generate activity/location pairs
    pairs = []
    for _ in range(count):
        activity = random.choice(ACTIVITIES)
        location = random.choice(LOCATIONS)
        pairs.append((activity, location))

    # Generate in batches
    print(f"\n[blue]Generating {count} records...[/]")
    
    good_records = []
    failed_attempts = 0
    llm_outputs = []
    
    # Process in chunks for better progress tracking
    chunk_size = 10
    generation_start = time.time()
    
    for i in track(range(0, count, chunk_size), description="Generating batches"):
        chunk_pairs = pairs[i:i+chunk_size]
        chunk_start = time.time()

        try:
            # Generate LLM outputs for this chunk with proper batching
            print(f"[debug] Generating chunk {i//chunk_size + 1} with {len(chunk_pairs)} pairs, batch_size={batch_size}")
            chunk_results = generate_batch(generator, chunk_pairs, batch_size)
            print(f"[debug] Got {len(chunk_results)} results from generate_batch")
            llm_outputs.extend(chunk_results)
            
            chunk_time = time.time() - chunk_start
            chunk_rate = len(chunk_results) / chunk_time if chunk_time > 0 else 0

            # Show progress
            if len(llm_outputs) % 20 == 0:
                elapsed = time.time() - generation_start
                rate = (len(llm_outputs) / elapsed) * 3600
                print(f"[dim]Generated {len(llm_outputs)} narratives - Rate: {rate:.0f}/hour[/]")
                print(f"[dim]Last chunk: {len(chunk_results)} records in {chunk_time:.1f}s ({chunk_rate:.1f} records/sec)[/]")

        except Exception as e:
            print(f"[red]Chunk generation failed: {e}[/]")
            failed_attempts += chunk_size

    # If we didn't get enough records, try to generate more
    if len(llm_outputs) < count * 0.8:  # If we got less than 80% of target
        print(f"\n[yellow]Warning: Only generated {len(llm_outputs)} records, target was {count}[/]")
        print(f"[blue]Attempting to generate additional records...[/]")
        
        additional_needed = count - len(llm_outputs)
        additional_pairs = []
        for _ in range(additional_needed):
            activity = random.choice(ACTIVITIES)
            location = random.choice(LOCATIONS)
            additional_pairs.append((activity, location))
        
        try:
            additional_results = generate_batch(generator, additional_pairs, batch_size)
            llm_outputs.extend(additional_results)
            print(f"[green]Generated {len(additional_results)} additional records[/]")
        except Exception as e:
            print(f"[red]Failed to generate additional records: {e}[/]")

    # Process and validate all outputs
    print(f"\n[blue]Processing and validating {len(llm_outputs)} generated records...[/]")
    
    validation_failures = []
    for i, llm_json in track(enumerate(llm_outputs),
                            description="Validating",
                            total=len(llm_outputs)):
        record, errors = process_record(llm_json)

        if record:
            good_records.append(record)
            if i < 3:  # Show first few
                print(f"[green]✓[/] Record {i+1}: {record['narrative'][:50]}...")
        else:
            failed_attempts += 1
            validation_failures.append((i, llm_json, errors))
            if i < 5 and errors:  # Show first few failures
                print(f"[yellow]✗[/] Record {i+1} failed: {errors}")
    
    # Show validation failure summary
    if validation_failures:
        print(f"\n[yellow]Validation Summary:[/]")
        print(f"  • Total generated: {len(llm_outputs)}")
        print(f"  • Validation failures: {len(validation_failures)}")
        print(f"  • Success rate: {(len(good_records)/len(llm_outputs)*100):.1f}%")
        
        # Show common validation errors
        error_types = {}
        for _, _, errors in validation_failures:
            for error in errors:
                error_types[error] = error_types.get(error, 0) + 1
        
        if error_types:
            print(f"\n[red]Common validation errors:[/]")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {error}: {count} occurrences")

    # Save results
    output_path = Path(outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if good_records:
        # Save as JSONL
        with open(outfile, 'w', encoding='utf-8') as f:
            for record in good_records:
                f.write(json.dumps(record, default=str) + '\n')

        # Also save as CSV
        csv_file = outfile.replace('.jsonl', '.csv')
        pd.DataFrame(good_records).to_csv(csv_file, index=False)

        # Calculate final statistics
        total_time = time.time() - start_time
        generation_rate = (len(good_records) / total_time) * 3600

        print(f"\n[bold green]✓ Generation Complete![/]")
        print(f"[green]Results:[/]")
        print(f"  • Valid records: {len(good_records)}/{count}")
        print(f"  • Failed validations: {failed_attempts}")
        print(f"  • Time taken: {total_time/60:.1f} minutes")
        print(f"  • Generation rate: {generation_rate:.0f} records/hour")
        print(f"  • Saved to: {outfile} and {csv_file}")

        if total_time < 1800:  # Under 30 minutes
            print(f"\n[bold green]🎉 Target achieved! Generated {len(good_records)} records in under 30 minutes![/]")
        else:
            print(f"\n[yellow]⚠ Target not achieved. Consider these optimizations:[/]")
            print(f"  • Use 'mistral-7b-fast' instead of 'mistral-7b' for 2-3x speed improvement")
            print(f"  • Try 'tinyllama' for fastest generation (lower quality)")
            print(f"  • Use 'phi-2' for balanced speed/quality")
            print(f"  • Increase batch_size if you have sufficient VRAM")
            print(f"  • Current generation rate: {generation_rate:.0f} records/hour")
            print(f"  • Estimated time for 100 records: {(100/generation_rate)*60:.1f} minutes")
    else:
        print(f"\n[bold red]✗ No valid records generated![/]")
        print(f"Failed attempts: {failed_attempts}")

    return len(good_records)


def benchmark_models():
    """Benchmark different model configurations"""
    configs = {
        "tinyllama": "TinyLlama (1.1B) - Fastest",
        "phi-2": "Microsoft Phi-2 (2.7B) - Balanced",
        "mistral-7b-fast": "Mistral 7B (8-bit) - Recommended",
        "mistral-7b": "Mistral 7B (fp16) - Best Quality"
    }

    print("[bold cyan]Model Benchmark for RTX 4070[/]")
    print("Testing 10 records with each model...\n")

    for config, description in configs.items():
        print(f"[cyan]Testing {description}...[/]")

        try:
            start = time.time()
            result = main(
                count=10,
                outfile=f"benchmark_{config}.jsonl",
                model_config=config,
                batch_size=2
            )
            elapsed = time.time() - start

            rate = (result / elapsed) * 3600 if elapsed > 0 else 0
            print(f"  → Generated {result}/10 in {elapsed:.1f}s")
            print(f"  → Estimated rate: {rate:.0f} records/hour")
            print(f"  → Estimated time for 100: {(100/rate)*60:.1f} minutes\n" if rate > 0 else "\n")

        except Exception as e:
            print(f"  [red]✗ Failed: {e}[/]\n")


if __name__ == "__main__":
    # Simple command line handling without typer decorators
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark-models":
        benchmark_models()
    else:
        # Use typer for main command
        typer.run(main)


# ver 5
'''
import random
import typer
import pandas as pd
from rich import print
from pydantic import ValidationError
from run_llm import generate_batch
from faker_inject import inject_fields
from qa_rules import validate_rules
from rich.progress import track


ACTIVITIES = [
    "reading session", "shopping trip", "community art class",
    "physio appointment", "swimming", "grocery run"
]
LOCATIONS = [
    "public library", "Westfield Mall", "community centre",
    "local pool", "physiotherapy clinic"
]

#app = typer.Typer(add_completion=False)

#@app.command()
def main(
    count:      int    = 50,
    batch_size: int    = 5,
    outfile:    str    = "incident_records.jsonl"
):
    """
       批量生成并验证数据，每 batch_size 条一次 LLM 调用。
       """
    good = []
    acts, locs = [], []
    total = 0

    print(f"[blue]Generating {count} records in batches of {batch_size} via LLM...[/]")

    for i in track(range(count), description="LLM batches"):
        acts.append(random.choice(ACTIVITIES))
        locs.append(random.choice(LOCATIONS))

        # 达到一个批次，或最后一批
        if len(acts) == batch_size or i == count - 1:
            try:
                batch_outputs = generate_batch(acts, locs)
            except Exception as e:
                print(f"[red]✗[/] Batch call failed: {e}")
                acts.clear();
                locs.clear()
                continue

            for json_rec in batch_outputs:
                try:
                    rec = inject_fields(json_rec)
                except ValidationError:
                    # 跳过不符合 schema 的记录
                    continue

                if not validate_rules(rec):
                    good.append(rec.dict())


            total += len(acts)
            print(f"[green]✓[/] Batch generated {len(acts)} (total {total}/{count})")
            acts.clear()
            locs.clear()

    if good:
        df = pd.DataFrame(good)
        df.to_json(outfile, orient="records", lines=True)
        csv = outfile.replace(".jsonl", ".csv")
        df.to_csv(csv, index=False)
        print(f"[bold green]✓ Saved {len(good)} valid records → {outfile}")
        print(f"[bold green]✓ CSV also saved → {csv}")
    else:
        print("[red]✗ No valid records generated![/]")

if __name__ == "__main__":
    typer.run(main)
'''

# ver 4
'''
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
'''