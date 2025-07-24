from faker import Faker
from datetime import datetime, timedelta
from data_schema import IncidentRecord

fake = Faker("en_AU")

def inject_fields(llm_json: dict) -> IncidentRecord:
    """Augment LLM output with Faker‑generated factual fields and validate."""
    record = {
        **llm_json,
        "date": fake.date_between("-30d", "today"),
        "name": fake.name(),
        "activity": llm_json.get("activity", "community outing"),
        "location": llm_json.get("location", fake.city()),
    }
    # randomise realistic start time if not present
    if "start_time" not in llm_json:
        hour = fake.random_int(9, 17)
        record["start_time"] = f"{hour:02d}:{fake.random_int(0,59):02d}"
    # ensure duration consistent
    record["duration_minutes"] = llm_json.get("duration_minutes", fake.random_int(20, 90))
    return IncidentRecord.parse_obj(record)
