MODEL_PROMPTS = {
    "mistralai/Mistral-7B-Instruct-v0.3":  You are a professional support worker writing an incident report. Create a realistic incident scenario during {activity} at {location}.

Generate a JSON object with a detailed narrative describing what happened, including specific actions you took and contributing factors you observed. The narrative should be 120-180 characters and describe a realistic support scenario.

JSON format:
```json
{{
  "narrative": "[Generate a realistic 120-180 character description of what happened during this activity, including what actions were taken and what factors contributed to the situation]",
  "start_time": "[HH:MM format between 09:00-17:00]",
  "duration_minutes": [number between 120-180],
  "participation": "[participated/refused/complained or empty string]",
  "actions_taken": ["[specific action 1]", "[specific action 2]"],
  "contributing_factors": ["[specific factor 1]", "[specific factor 2]"],
  "productivity_level": [number 1-5],
  "engagement_level": [number 1-3],
  "activity": "{activity}",
  "location": "{location}"
}}
}