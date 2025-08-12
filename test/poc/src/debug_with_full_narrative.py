#!/usr/bin/env python
"""Debug script to see the full narrative before truncation"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path.cwd()))

from run_llm import generate as llm_generate


def show_full_narrative():
    """Generate content and show the full narrative before processing"""
    print("=== Testing Full Narrative Generation ===\n")

    try:
        # Generate LLM content
        print("Generating LLM content...")
        llm_json = llm_generate(activity="reading session", location="public library")

        print(f"\n=== FULL LLM OUTPUT ===")
        for key, value in llm_json.items():
            print(f"{key}: {value}")

        print(f"\n=== NARRATIVE ANALYSIS ===")
        narrative = llm_json.get("narrative", "")
        print(f"Length: {len(narrative)} characters")
        print(f"Full narrative: '{narrative}'")

        # Check for validation phrases
        narrative_lower = narrative.lower()
        print(f"\nValidation checks:")
        print(f"  Contains 'actions taken': {'actions taken' in narrative_lower}")
        print(f"  Contains 'action': {'action' in narrative_lower}")
        print(f"  Contains 'factor': {'factor' in narrative_lower}")
        print(f"  Contains 'factors': {'factors' in narrative_lower}")

        # Show where truncation would occur
        if len(narrative) > 180:
            print(f"\nTruncation at 180 chars:")
            print(f"  Truncated: '{narrative[:180]}'")
            print(f"  Lost part: '{narrative[180:]}'")

        return narrative

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    show_full_narrative()