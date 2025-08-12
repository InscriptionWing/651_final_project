#!/usr/bin/env python
"""Test the specific case that was failing"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path.cwd()))

from faker_inject import inject_fields
from qa_rules import validate_rules


def test_problematic_narrative():
    """Test the specific narrative that was failing"""

    # This is the LLM output that was causing issues
    problematic_llm_json = {
        "narrative": "Assisted client with cognitive impairment during shopping trip at community centre. Client became agitated in a crowded store due to sensory overload (contributing factors). Calmed client and relocated to quieter area.",
        "start_time": "14:30",
        "duration_minutes": 150,
        "participation": "participated",
        "actions_taken": ["Calmed client", "Relocated to quieter area"],
        "contributing_factors": ["Sensory overload", "Crowded environment"],
        "productivity_level": 3,
        "engagement_level": 2,
        "activity": "shopping trip",
        "location": "community centre"
    }

    print("=== Testing Problematic Case ===")
    print(f"Original narrative: '{problematic_llm_json['narrative']}'")
    print(f"Length: {len(problematic_llm_json['narrative'])}")

    # Check validation phrases in original
    orig_lower = problematic_llm_json['narrative'].lower()
    print(f"\nOriginal validation check:")
    print(f"  Has 'actions taken': {'actions taken' in orig_lower}")
    print(f"  Has 'action': {'action' in orig_lower}")
    print(f"  Has 'factor': {'factor' in orig_lower}")
    print(f"  Has 'factors': {'factors' in orig_lower}")

    try:
        # Process through inject_fields
        record = inject_fields(problematic_llm_json)

        print(f"\nAfter processing:")
        print(f"Final narrative: '{record.narrative}'")
        print(f"Length: {len(record.narrative)}")

        # Check validation phrases in final
        final_lower = record.narrative.lower()
        print(f"\nFinal validation check:")
        print(f"  Has 'actions taken': {'actions taken' in final_lower}")
        print(f"  Has 'action': {'action' in final_lower}")
        print(f"  Has 'factor': {'factor' in final_lower}")
        print(f"  Has 'factors': {'factors' in final_lower}")

        # Run validation
        errors = validate_rules(record)
        print(f"\nValidation result:")
        print(f"  Errors: {errors}")
        print(f"  Valid: {len(errors) == 0}")

        return len(errors) == 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test other edge cases"""

    test_cases = [
        {
            "name": "Already perfect narrative",
            "narrative": "Client participated in reading session. Actions taken included providing assistance. Contributing factors included lighting and attention span.",
            "expected_valid": True
        },
        {
            "name": "Missing both phrases",
            "narrative": "Client engaged well during the session. Support was provided throughout the activity.",
            "expected_valid": True  # Should be made valid by injection
        },
        {
            "name": "Has factors but no actions",
            "narrative": "Session went well but environmental factors affected the outcome significantly.",
            "expected_valid": True  # Should be made valid by injection
        },
        {
            "name": "Has actions but no factors",
            "narrative": "Support worker took appropriate actions to assist the client during the session.",
            "expected_valid": True  # Should be made valid by injection
        }
    ]

    print("\n=== Testing Edge Cases ===")

    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i + 1}: {test_case['name']}")

        llm_json = {
            "narrative": test_case["narrative"],
            "start_time": "14:30",
            "duration_minutes": 150,
            "participation": "participated",
            "actions_taken": ["Provided support", "Monitored situation"],
            "contributing_factors": ["Environmental factors", "Personal factors"],
            "productivity_level": 3,
            "engagement_level": 2,
            "activity": "reading session",
            "location": "public library"
        }

        try:
            record = inject_fields(llm_json)
            errors = validate_rules(record)
            is_valid = len(errors) == 0

            print(f"  Original: '{test_case['narrative'][:60]}...'")
            print(f"  Final: '{record.narrative[:60]}...'")
            print(f"  Valid: {is_valid} (expected: {test_case['expected_valid']})")

            if is_valid != test_case['expected_valid']:
                print(f"  ❌ UNEXPECTED RESULT!")
                print(f"  Errors: {errors}")
            else:
                print(f"  ✅ Expected result")

        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    # Test the specific problematic case
    success = test_problematic_narrative()

    # Test other edge cases
    test_edge_cases()

    if success:
        print(f"\n🎉 Problematic case now works!")
    else:
        print(f"\n❌ Still having issues with the problematic case")


if __name__ == "__main__":
    main()