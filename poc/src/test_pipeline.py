#!/usr/bin/env python

print("Testing pipeline components...")

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    from run_llm import generate as llm_generate

    print("✓ run_llm imported successfully")

    from faker_inject import inject_fields

    print("✓ faker_inject imported successfully")

    from qa_rules import validate_rules

    print("✓ qa_rules imported successfully")

except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check environment
print("\n2. Testing environment...")
try:
    import os
    from dotenv import load_dotenv

    load_dotenv()

    model = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")
    print(f"✓ Model set to: {model}")

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("✓ Created data directory")
    else:
        print("✓ Data directory exists")

except Exception as e:
    print(f"✗ Environment setup failed: {e}")

# Test 3: Single generation
print("\n3. Testing single record generation...")
try:
    result = llm_generate(activity="reading session", location="public library")
    print(f"✓ LLM generation successful")
    print(f"  Result type: {type(result)}")
    print(f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

    # Test injection
    record = inject_fields(result)
    print(f"✓ Field injection successful")
    print(f"  Record type: {type(record)}")

    # Test validation
    errors = validate_rules(record)
    print(f"✓ Validation complete")
    print(f"  Errors: {errors}")
    print(f"  Valid: {len(errors) == 0}")

    if len(errors) == 0:
        print("\n✓ Full pipeline test PASSED")
        print("You can now run: python merge_pipeline.py --count 5 --outfile data/test.jsonl")
    else:
        print(f"\n⚠ Pipeline works but has validation issues: {errors}")

except Exception as e:
    print(f"✗ Pipeline test failed: {e}")
    import traceback

    traceback.print_exc()