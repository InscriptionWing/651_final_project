from data_schema import IncidentRecord
from typing import List

def validate_rules(rec: IncidentRecord) -> list[str]:
    #errors = []
    '''
    if "actions taken" not in rec.narrative.lower():
        errors.append("Narrative missing 'actions taken' phrase")
    if "factor" not in rec.narrative.lower():
        errors.append("Narrative missing contributing factors mention")
    return errors
    '''

    """
        Validate business rules for incident records.
        Returns a list of error messages. Empty list means valid.
    """
    errors = []

    # Check if narrative contains required phrases (case insensitive)
    narrative_lower = rec.narrative.lower()

    # Rule 1: Must mention actions taken
    if "actions taken" not in narrative_lower and "action" not in narrative_lower:
        errors.append("Narrative missing 'actions taken' or 'action' mention")

    # Rule 2: Must mention contributing factors
    if "factor" not in narrative_lower and "factors" not in narrative_lower:
        errors.append("Narrative missing contributing factors mention")

    # Optional: Additional validation rules
    # Rule 3: Narrative should have reasonable length
    if len(rec.narrative) < 50:
        errors.append("Narrative too short (less than 50 characters)")

    # Rule 4: Check if required fields are populated
    if not rec.actions_taken or len(rec.actions_taken) == 0:
        errors.append("No actions_taken specified")

    if not rec.contributing_factors or len(rec.contributing_factors) == 0:
        errors.append("No contributing_factors specified")

    return errors
