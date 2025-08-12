#!/usr/bin/env python
"""Debug script to test the validation logic"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path.cwd()))

from run_llm import generate as llm_generate
from faker_inject import inject_fields
from qa_rules import validate_rules


def test_single_record():
    """Test generating and validating a single record"""
    print("=== Testing Single Record Generation ===\n")

    try:
        # Generate LLM content
        print("1. Generating LLM content...")
        llm_json = llm_generate(activity="reading session", location="public library")
        print(f"✓ LLM generated: {type(llm_json)}")
        print(f"Keys: {list(llm_json.keys())}")
        print(f"Narrative: '{llm_json.get('narrative', 'MISSING')}'")

        # Inject fields
        print("\n2. Injecting additional fields...")
        record = inject_fields(llm_json)
        print(f"✓ Record created: {type(record)}")
        print(f"Final narrative: '{record.narrative}'")
        print(f"Actions taken: {record.actions_taken}")
        print(f"Contributing factors: {record.contributing_factors}")

        # Validate
        print("\n3. Validating record...")
        errors = validate_rules(record)
        print(f"Validation errors: {errors}")
        print(f"Is valid: {len(errors) == 0}")

        # Show what validation is checking for
        print("\n4. Validation details...")
        narrative_lower = record.narrative.lower()
        print(f"Contains 'actions taken': {'actions taken' in narrative_lower}")
        print(f"Contains 'action': {'action' in narrative_lower}")
        print(f"Contains 'factor': {'factor' in narrative_lower}")
        print(f"Contains 'factors': {'factors' in narrative_lower}")
        print(f"Narrative length: {len(record.narrative)}")

        return len(errors) == 0

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_records(count=5):
    """Test generating multiple records"""
    print(f"\n=== Testing {count} Records ===\n")

    valid_count = 0
    activities = ["reading session", "shopping trip", "swimming"]
    locations = ["public library", "community centre", "local pool"]

    for i in range(count):
        print(f"Record {i + 1}:")
        try:
            # Pick random activity and location
            import random
            activity = random.choice(activities)
            location = random.choice(locations)

            # Generate and validate
            llm_json = llm_generate(activity=activity, location=location)
            record = inject_fields(llm_json)
            errors = validate_rules(record)

            if len(errors) == 0:
                print(f"  ✓ Valid")
                valid_count += 1
            else:
                print(f"  ✗ Invalid: {errors}")
                print(f"    Narrative: '{record.narrative[:100]}...'")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nSummary: {valid_count}/{count} records valid")
    return valid_count


def main():
    print("=== Validation Debug Script ===")

    # Test single record
    single_success = test_single_record()

    # Test multiple records
    if single_success:
        valid_count = test_multiple_records(5)

        if valid_count > 0:
            print(f"\n🎉 Validation is working! {valid_count} valid records generated.")
            print("The pipeline should work now.")
        else:
            print(f"\n❌ No valid records generated. Check validation rules.")
    else:
        print(f"\n❌ Single record test failed. Check the error above.")


if __name__ == "__main__":
    main()