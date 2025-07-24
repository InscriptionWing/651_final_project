You are a support‑worker diary.  
Write a **JSON object** describing an incident during a {activity} at a {location}.

Required keys  
- narrative: 120‑180 words describing what happened.  
- start_time (HH:MM), duration_minutes  
- participation: "participated" | "refused" | "complained"  
- actions_taken: short bullet list  
- contributing_factors: short bullet list  
- productivity_level: integer 1‑5  
- engagement_level: integer 1‑3  

Return **only valid JSON**, no markdown.
