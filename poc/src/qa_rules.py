from data_schema import IncidentRecord

def validate_rules(rec: IncidentRecord) -> list[str]:
    errors = []
    if "actions taken" not in rec.narrative.lower():
        errors.append("Narrative missing 'actions taken' phrase")
    if "factor" not in rec.narrative.lower():
        errors.append("Narrative missing contributing factors mention")
    return errors
