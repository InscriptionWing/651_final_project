from faker import Faker
from datetime import datetime, timedelta
from data_schema import IncidentRecord
import re

fake = Faker("en_AU")


def inject_fields(llm_json: dict) -> IncidentRecord:
    """Augment and sanitize LLM output, ensuring schema compliance."""
    record = {}

    # 1. Narrative: pad or truncate to 120–180 characters, but ensure it contains required phrases
    narr = llm_json.get("narrative", "")
    narr = " ".join(narr.split())

    print(f"[dim]Original narrative ({len(narr)} chars): {narr}[/]")

    # Check if the narrative already contains required validation phrases
    narr_lower = narr.lower()
    has_actions = "actions taken" in narr_lower or "action" in narr_lower
    has_factors = "factor" in narr_lower or "factors" in narr_lower

    print(f"[dim]Has actions: {has_actions}, Has factors: {has_factors}[/]")

    # FIRST: Handle length constraints - do this BEFORE adding phrases to avoid mid-sentence cuts
    if len(narr) > 180:
        # Truncate carefully at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', narr)
        truncated = ""

        for sentence in sentences:
            test_length = len(truncated + " " + sentence if truncated else sentence)
            if test_length <= 180:
                truncated = truncated + " " + sentence if truncated else sentence
            else:
                break

        if truncated and len(truncated) >= 100:  # Keep if reasonably long
            narr = truncated
        else:
            # Fallback: truncate at word boundary near 177 chars
            words = narr.split()
            truncated = ""
            for word in words:
                test_length = len(truncated + " " + word if truncated else word)
                if test_length <= 177:
                    truncated = truncated + " " + word if truncated else word
                else:
                    break
            narr = truncated + "..." if truncated else narr[:177] + "..."

    # Re-check phrases after truncation
    narr_lower = narr.lower()
    has_actions = "actions taken" in narr_lower or "action" in narr_lower
    has_factors = "factor" in narr_lower or "factors" in narr_lower

    print(f"[dim]After truncation - Has actions: {has_actions}, Has factors: {has_factors}[/]")

    # SECOND: Add missing phrases if needed
    if not has_actions and not has_factors:
        # Both missing - add both
        addition = " Actions taken to address contributing factors."
        if len(narr) + len(addition) <= 180:
            narr += addition
        else:
            # Replace some text to fit both phrases
            available_space = 180 - len(" Actions taken addressed factors.")
            if available_space > 50:  # Keep some original content
                narr = narr[:available_space] + " Actions taken addressed factors."
            else:
                narr = narr[:120] + " Actions taken addressed factors."

    elif not has_actions:
        # Only actions missing - need to add actions but preserve existing factors
        if "factor" in narr_lower or "factors" in narr_lower:
            # We have factors, just need actions - try to add naturally
            addition = " Actions taken addressed these issues."
            if len(narr) + len(addition) <= 180:
                narr += addition
            else:
                # Replace end with action phrase that preserves factors context
                # Find a good insertion point that keeps factors mention
                factor_pos = max(narr_lower.find("factor"), narr_lower.find("factors"))
                if factor_pos >= 0:
                    # Keep text up to and including factors mention, then add actions
                    factor_end = narr_lower.find(")", factor_pos)  # Look for closing paren
                    if factor_end >= 0:
                        keep_until = factor_end + 1
                        remaining_space = 180 - keep_until - len(" Actions taken provided support.")
                        if remaining_space > 0:
                            narr = narr[:keep_until] + " Actions taken provided support."
                        else:
                            # Not enough space, rebuild with both phrases
                            available_space = 180 - len(" Actions taken addressed contributing factors.")
                            narr = narr[:available_space] + " Actions taken addressed contributing factors."
                    else:
                        # No closing paren, just append
                        available_space = 180 - len(" Actions taken provided support.")
                        narr = narr[:available_space] + " Actions taken provided support."
                else:
                    # Fallback
                    available_space = 180 - len(" Actions taken addressed factors.")
                    narr = narr[:available_space] + " Actions taken addressed factors."
        else:
            # No factors either, add both
            available_space = 180 - len(" Actions taken addressed factors.")
            narr = narr[:available_space] + " Actions taken addressed factors."

    elif not has_factors:
        # Only factors missing
        addition = " Contributing factors were considered."
        if len(narr) + len(addition) <= 180:
            narr += addition
        else:
            # Try to insert more naturally
            if "action" in narr_lower:
                # Insert factors before actions
                narr = re.sub(r'(actions?)\s+(taken|provided)', r'factors influenced outcomes. \1 \2', narr, count=1,
                              flags=re.IGNORECASE)
                if len(narr) > 180:
                    narr = narr[:180]
            else:
                # Fallback: replace end with factors phrase
                available_space = 180 - len(" Contributing factors noted.")
                narr = narr[:available_space] + " Contributing factors noted."

    # THIRD: Ensure minimum length
    if len(narr) < 120:
        # Pad to minimum length
        padding_options = [
            " The support worker ensured client comfort.",
            " Additional support was provided as needed.",
            " The situation was monitored throughout.",
            " Documentation was completed appropriately."
        ]

        for padding in padding_options:
            if len(narr) + len(padding) <= 180 and len(narr) + len(padding) >= 120:
                narr += padding
                break

        # If still too short, add generic padding
        while len(narr) < 120:
            remaining = 120 - len(narr)
            if remaining > 30:
                narr += " The support worker ensured client comfort."
            else:
                narr += " Additional support provided."

            if len(narr) > 180:
                narr = narr[:180]
                break

    # FINAL: Ensure we still have both required phrases after all processing
    final_lower = narr.lower()
    final_has_actions = "actions taken" in final_lower or "action" in final_lower
    final_has_factors = "factor" in final_lower or "factors" in final_lower

    if not final_has_actions or not final_has_factors:
        # Emergency fix: rebuild narrative with required phrases
        base_length = min(100, len(narr))
        base_text = narr[:base_length].rsplit(' ', 1)[0]  # Cut at word boundary

        if not final_has_actions and not final_has_factors:
            narr = base_text + " Actions taken addressed contributing factors appropriately."
        elif not final_has_actions:
            narr = base_text + " Actions taken provided necessary support."
        elif not final_has_factors:
            narr = base_text + " Contributing factors were carefully considered."

        # Ensure length constraints
        if len(narr) > 180:
            narr = narr[:177] + "..."
        elif len(narr) < 120:
            narr += " Additional documentation completed."

    final_check = narr.lower()
    if "action" not in final_check:
        # Force add it in a way that won't get truncated
        if len(narr) > 150:
            narr = narr[:120] + " Actions taken to address factors."
        else:
            narr += " Actions taken were appropriate."

    record["narrative"] = narr
    print(f"[dim]Final narrative ({len(narr)} chars): {narr}[/]")

    # 2. Start time: parse AM/PM or fallback to random between 09:00–17:00
    st_raw = llm_json.get("start_time", "")
    if re.search(r"\d{1,2}:\d{2}\s*[AP]M", st_raw, re.IGNORECASE):
        try:
            dt = datetime.strptime(st_raw.strip().upper(), "%I:%M %p")
            record["start_time"] = dt.time()
        except ValueError:
            # Fallback if parsing fails
            hour = fake.random_int(9, 17)
            minute = fake.random_int(0, 59)
            record["start_time"] = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
    elif re.match(r"\d{1,2}:\d{2}", st_raw):
        try:
            record["start_time"] = datetime.strptime(st_raw, "%H:%M").time()
        except ValueError:
            hour = fake.random_int(9, 17)
            minute = fake.random_int(0, 59)
            record["start_time"] = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
    else:
        hour = fake.random_int(9, 17)
        minute = fake.random_int(0, 59)
        record["start_time"] = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()

    # 3. Duration: clamp to 120–180
    dur = llm_json.get("duration_minutes", fake.random_int(120, 180))
    if not isinstance(dur, int) or dur < 120 or dur > 180:
        dur = fake.random_int(120, 180)
    record["duration_minutes"] = dur

    # 4. Participation normalization
    part_raw = llm_json.get("participation", "")
    part_lower = part_raw.lower()
    if "particip" in part_lower:
        record["participation"] = "participated"
    elif "refus" in part_lower:
        record["participation"] = "refused"
    elif "complain" in part_lower:
        record["participation"] = "complained"
    else:
        record["participation"] = ""

    # 5. Actions taken: ensure at least two items
    ats = llm_json.get("actions_taken") or []
    if not isinstance(ats, list):
        ats = [str(ats)]
    while len(ats) < 2:
        ats.append("Monitored the situation")
    record["actions_taken"] = ats[:2]

    # 6. Contributing factors: ensure at least two items
    cfs = llm_json.get("contributing_factors") or []
    if not isinstance(cfs, list):
        cfs = [str(cfs)]
    while len(cfs) < 2:
        cfs.append("Unknown factor")
    record["contributing_factors"] = cfs[:2]

    # 7. Productivity level: clamp to 1–5
    prod = llm_json.get("productivity_level", fake.random_int(1, 5))
    if not isinstance(prod, int) or prod < 1 or prod > 5:
        prod = fake.random_int(1, 5)
    record["productivity_level"] = prod

    # 8. Engagement level: clamp to 1–3
    eng = llm_json.get("engagement_level", fake.random_int(1, 3))
    if not isinstance(eng, int) or eng < 1 or eng > 3:
        eng = fake.random_int(1, 3)
    record["engagement_level"] = eng

    # 9. Activity & location from LLM or defaults
    record["activity"] = llm_json.get("activity", "community outing")
    record["location"] = llm_json.get("location", fake.city())

    # 10. Factual fields: date & name
    record["date"] = fake.date_between("-30d", "today")
    record["name"] = fake.name()

    # Validate against schema (may raise ValidationError)
    return IncidentRecord.parse_obj(record)

'''
fake = Faker("en_AU")

def inject_fields(llm_json: dict) -> IncidentRecord:
    """Augment and sanitize LLM output, ensuring schema compliance."""
    record = {}

    # 1. Narrative: pad or truncate to 120–180 characters
    narr = llm_json.get("narrative", "")
    narr = " ".join(narr.split())

    print(f"[dim]Original narrative ({len(narr)} chars): {narr}[/]")

    # Check if the narrative already contains required validation phrases
    narr_lower = narr.lower()
    has_actions = "actions taken" in narr_lower or "action" in narr_lower
    has_factors = "factor" in narr_lower or "factors" in narr_lower

    # If narrative is missing required phrases, add them before truncating
    if not has_actions:
        # Insert "actions taken" phrase naturally
        if len(narr) < 120:
            narr += " Actions taken included providing support and monitoring the situation."
        else:
            # Try to insert it more naturally in the middle
            words = narr.split()
            middle_point = len(words) // 2
            words.insert(middle_point, "Actions taken included appropriate interventions.")
            narr = " ".join(words)

    if not has_factors:
        # Insert "factors" phrase naturally
        if len(narr) < 140:
            narr += " Contributing factors included environmental and personal elements."
        else:
            # Insert at appropriate location
            if "actions taken" in narr.lower():
                # Add factors after actions
                narr = narr.replace("Actions taken", "Actions taken to address contributing factors")
            else:
                # Add factors mention
                words = narr.split()
                insert_point = min(len(words) - 10, len(words) // 3 * 2)  # Near end but not at end
                words.insert(insert_point, "Various factors contributed to the situation.")
                narr = " ".join(words)

    if len(narr) < 120:
        pad_text = " The support worker ensured client comfort."
        while len(narr) < 120:
            narr += pad_text
            if len(narr) > 180:
                break
        narr = narr[:180]
    elif len(narr) > 180:
        #narr = narr[:180]
        # Truncate carefully - try to preserve validation phrases
        # Find the last complete sentence that fits in 180 chars
        sentences = narr.split('. ')
        truncated = ""

        for sentence in sentences:
            test_length = len(truncated + sentence + '. ')
            if test_length <= 180:
                if truncated:
                    truncated += '. ' + sentence
                else:
                    truncated = sentence
            else:
                break

        if truncated and len(truncated) >= 120:
            narr = truncated + '.' if not truncated.endswith('.') else truncated
        else:
            # Fallback: simple truncation but ensure we keep validation phrases
            narr = narr[:177] + "..."

            # Double-check we still have required phrases after truncation
            if "action" not in narr.lower():
                # Emergency fix: replace some text to include action
                narr = narr[:100] + " Actions taken included support. Contributing factors noted..."[:80]
            elif "factor" not in narr.lower():
                # Emergency fix: replace some text to include factors
                narr = narr[:100] + " Actions taken addressed factors. Environmental factors considered..."[:80]

    record["narrative"] = narr
    print(f"[dim]Final narrative ({len(narr)} chars): {narr}[/]")

    # 2. Start time: parse AM/PM or fallback to random between 09:00–17:00
    st_raw = llm_json.get("start_time", "")
    if re.search(r"\d{1,2}:\d{2}\s*[AP]M", st_raw, re.IGNORECASE):
        # dt = datetime.strptime(st_raw.strip().upper(), "%I:%M %p")
        # record["start_time"] = dt.strftime("%H:%M")
        try:
            dt = datetime.strptime(st_raw.strip().upper(), "%I:%M %p")
            record["start_time"] = dt.time()
        except ValueError:
            # Fallback if parsing fails
            hour = fake.random_int(9, 17)
            minute = fake.random_int(0, 59)
            record["start_time"] = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()

    elif re.match(r"\d{1,2}:\d{2}", st_raw):
        #record["start_time"] = st_raw
        try:
            record["start_time"] = datetime.strptime(st_raw, "%H:%M").time()
        except ValueError:
            hour = fake.random_int(9, 17)
            minute = fake.random_int(0, 59)
            record["start_time"] = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()

    else:
        hour = fake.random_int(9, 17)
        minute = fake.random_int(0, 59)
        record["start_time"] = f"{hour:02d}:{minute:02d}"

    # 3. Duration: clamp to 120–180
    dur = llm_json.get("duration_minutes", fake.random_int(120, 180))
    if not isinstance(dur, int) or dur < 120 or dur > 180:
        dur = fake.random_int(120, 180)
    record["duration_minutes"] = dur

    # 4. Participation normalization
    part_raw = llm_json.get("participation", "")
    part_lower = part_raw.lower()
    if "particip" in part_lower:
        record["participation"] = "participated"
    elif "refus" in part_lower:
        record["participation"] = "refused"
    elif "complain" in part_lower:
        record["participation"] = "complained"
    else:
        record["participation"] = ""

    # 5. Actions taken: ensure at least two items
    ats = llm_json.get("actions_taken") or []
    if not isinstance(ats, list):
        ats = [str(ats)]
    while len(ats) < 2:
        ats.append("Monitored the situation")
    record["actions_taken"] = ats[:2]

    # 6. Contributing factors: ensure at least two items
    cfs = llm_json.get("contributing_factors") or []
    if not isinstance(cfs, list):
        cfs = [str(cfs)]
    while len(cfs) < 2:
        cfs.append("Unknown factor")
    record["contributing_factors"] = cfs[:2]

    # 7. Productivity level: clamp to 1–5
    prod = llm_json.get("productivity_level", fake.random_int(1, 5))
    if not isinstance(prod, int) or prod < 1 or prod > 5:
        prod = fake.random_int(1, 5)
    record["productivity_level"] = prod

    # 8. Engagement level: clamp to 1–3
    eng = llm_json.get("engagement_level", fake.random_int(1, 3))
    if not isinstance(eng, int) or eng < 1 or eng > 3:
        eng = fake.random_int(1, 3)
    record["engagement_level"] = eng

    # 9. Activity & location from LLM or defaults
    record["activity"] = llm_json.get("activity", "community outing")
    record["location"] = llm_json.get("location", fake.city())

    # 10. Factual fields: date & name
    record["date"] = fake.date_between("-30d", "today")
    record["name"] = fake.name()

    # Validate against schema (may raise ValidationError)
    return IncidentRecord.parse_obj(record)
'''

'''
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
'''