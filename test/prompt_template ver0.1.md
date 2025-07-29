

You are a support‑worker diary.  
Generate a **single JSON object** describing an incident during a {activity} at a {location}.  
**Output must be pure JSON**, no extra text.

Required JSON schema (fill exactly):  
```json
{{
  "narrative": "String, 120–180 characters describing what happened",
  "start_time": "String, HH:MM between 09:00 and 17:00",
  "duration_minutes": number, 120–180,
  "participation": "participated" | "refused" | "complained" | "",
  "actions_taken": ["String action related to the scenario", "Another action"],
  "contributing_factors": ["String factor 1", "String factor 2"],
  "productivity_level": integer, 1–5,
  "engagement_level": integer, 1–3,
  "activity": "{activity}",
  "location": "{location}"
}}