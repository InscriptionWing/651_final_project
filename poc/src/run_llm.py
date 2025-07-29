MODEL_PROMPTS = {
    "mistralai/Mistral-7B-Instruct-v0.3": """<s>[INST] You are a professional support worker writing an incident report. Create a realistic incident scenario during {activity} at {location}.

Generate a JSON object with a detailed narrative describing what happened, including specific actions you took and contributing factors you observed. The narrative should be 120-180 characters and describe a realistic support scenario.

JSON format:
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
}}[/INST]"""
}

import os
import json
import re
import typer
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich import print
from pathlib import Path


def find_and_load_env():
    current_path = Path(__file__).parent

    # Check current directory first
    env_path = current_path / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[green]✓[/] Loaded .env from: {env_path}")
        return True

    # Check parent directory
    env_path = current_path.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[green]✓[/] Loaded .env from: {env_path}")
        return True

    # Check root directory (two levels up)
    env_path = current_path.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[green]✓[/] Loaded .env from: {env_path}")
        return True

    print(f"[yellow]⚠[/] No .env file found in current, parent, or root directories")
    return False


# Load environment variables
find_and_load_env()

# load_dotenv()
app = typer.Typer(add_completion=False)

# Global variables
tokenizer = None
model = None
generator = None


def load_model():
    """Load model with better configuration for narrative generation"""
    global tokenizer, model, generator

    if generator is not None:
        return generator

    MODEL_ID = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    HF_TOKEN = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    print(f"[blue]Loading model for narrative generation:[/] {MODEL_ID}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            token=HF_TOKEN  # Add this parameter
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,  # Increased for narrative generation
            do_sample=True,
            temperature=0.7,  # Higher temperature for more creative narratives
            top_p=0.9,  # Add nucleus sampling for better variety
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"[green]✓[/] Successfully loaded {MODEL_ID} for narrative generation")
        return generator

    except Exception as e:
        print(f"[red]✗[/] Failed to load {MODEL_ID}: {e}")
        return create_narrative_mock_generator()


def create_narrative_mock_generator():
    """Enhanced mock generator with realistic narrative variations"""
    print("[yellow]⚠ Using narrative mock generator[/]")

    # Realistic incident scenarios for different activities
    NARRATIVE_TEMPLATES = {
        "reading session": [
            "Client showed interest in adventure novels but became frustrated with small text. Provided magnifying glass and encouraged shorter sessions. Client's vision difficulties and lighting conditions contributed to challenges.",
            "During reading session, client selected cookbook and engaged well initially. Became restless after 45 minutes. Offered break and different seating. Time of day and attention span were key factors.",
            "Client participated in group reading activity. Initially shy but warmed up when discussing favorite characters. Facilitated introductions and provided encouragement. Social anxiety and group dynamics influenced participation.",
            "Client chose mystery novel and read quietly for extended period. Asked several questions about plot. Provided explanations and praise for engagement. Client's curiosity and genre preference enhanced experience.",
        ],
        "shopping trip": [
            "Client managed shopping list well but became overwhelmed in crowded aisles. Guided to quieter sections and helped prioritize items. Store crowds and sensory overload were contributing factors.",
            "During grocery shopping, client forgot wallet and became distressed. Reassured client and contacted family for support. Memory challenges and anxiety affected the outing experience.",
            "Client enjoyed selecting fresh produce and comparing prices. Needed assistance reading small labels. Provided reading support and praised decision-making skills. Visual impairment and independence goals were factors.",
            "Shopping trip went smoothly until checkout queues caused frustration. Found shorter line and used calming techniques. Wait times and impatience were main contributing elements.",
        ],
        "community art class": [
            "Client enthusiastically started watercolor painting but struggled with brush control. Demonstrated techniques and provided adaptive brushes. Fine motor skills and equipment suitability affected performance.",
            "During art session, client felt intimidated by others' work. Encouraged personal expression and highlighted unique style. Comparison anxiety and self-confidence were significant factors.",
            "Client created beautiful collage but spilled paint on clothing. Helped clean up and reassured about accidents. Coordination challenges and material management influenced the session.",
            "Art class progressed well with client showing creativity in sculpture. Needed support with tool usage. Provided guidance and celebrated achievements. Motor skills and artistic confidence were key.",
        ],
        "physio appointment": [
            "Client arrived early and showed motivation for exercises. Complained of soreness midway through session. Adjusted intensity and provided rest breaks. Pain levels and exercise tolerance were factors.",
            "During physiotherapy, client struggled with balance exercises. Provided additional support and modified routine. Inner ear issues and fear of falling contributed to difficulties.",
            "Client completed prescribed exercises but rushed through repetitions. Encouraged slower pace and proper form. Impatience and understanding of technique importance were influences.",
            "Physiotherapy session successful with client achieving new range of motion goals. Celebrated progress and planned next steps. Consistent attendance and effort were contributing positives.",
        ],
        "swimming": [
            "Client enjoyed pool time but hesitated at deep end. Stayed in shallow area and provided reassurance. Previous negative experience and water depth anxiety were key factors.",
            "During swimming session, client showed improved stroke technique. Needed breaks every few lengths. Provided encouragement and paced activity. Fitness level and breathing control influenced performance.",
            "Client participated in water aerobics class with enthusiasm. Required assistance entering pool. Helped with mobility and safety precautions. Physical limitations and pool access were considerations.",
            "Swimming went well until other pool users became noisy. Moved to quieter lane and continued exercises. Sensory sensitivity and environmental distractions affected concentration.",
        ],
        "grocery run": [
            "Client managed small shopping list independently but forgot milk. Returned to dairy section together and completed task. Memory aids and routine establishment were important factors.",
            "During grocery run, client became confused by store layout changes. Provided navigation help and found familiar products. Store modifications and adaptation challenges influenced the trip.",
            "Client selected healthy options and stayed within budget constraints. Praised good choices and mathematical skills. Nutritional awareness and financial management were positive contributors.",
            "Grocery shopping complicated by client's mobility device in narrow aisles. Found accessible routes and assisted with reaching items. Store design and physical accessibility were factors.",
        ]
    }

    def mock_generate(prompt, **kwargs):
        import random

        # Extract activity and location
        activity = "reading session"
        location = "public library"

        # Extract from prompt
        for act in NARRATIVE_TEMPLATES.keys():
            if act in prompt.lower():
                activity = act
                break

        locations = ["public library", "westfield mall", "community centre", "local pool", "physiotherapy clinic"]
        for loc in locations:
            if loc in prompt.lower():
                location = loc
                break

        # Get realistic narrative for activity
        if activity in NARRATIVE_TEMPLATES:
            narrative = random.choice(NARRATIVE_TEMPLATES[activity])
        else:
            narrative = f"Client engaged in {activity} with varying levels of participation. Support provided throughout session with attention to individual needs. Environmental and personal factors influenced the experience."

        # Ensure narrative is within character limit
        if len(narrative) < 120:
            narrative += " Additional observations and support strategies were documented for future reference."
        if len(narrative) > 180:
            narrative = narrative[:177] + "..."

        # Generate realistic supporting data
        participation_options = ["participated", "refused", "complained", ""]

        # Activity-specific actions
        action_sets = {
            "reading session": [
                ["Provided reading assistance", "Adjusted lighting conditions"],
                ["Selected appropriate materials", "Encouraged comprehension discussion"],
                ["Offered reading breaks", "Supported text navigation"]
            ],
            "shopping trip": [
                ["Assisted with navigation", "Managed budget tracking"],
                ["Provided decision support", "Ensured safety protocols"],
                ["Facilitated item selection", "Supported payment process"]
            ],
            "community art class": [
                ["Demonstrated techniques", "Provided adaptive tools"],
                ["Encouraged creative expression", "Assisted with material management"],
                ["Facilitated peer interaction", "Supported cleanup process"]
            ],
            "physio appointment": [
                ["Monitored exercise form", "Adjusted intensity levels"],
                ["Provided mobility support", "Encouraged proper technique"],
                ["Documented progress notes", "Coordinated with therapist"]
            ],
            "swimming": [
                ["Ensured water safety", "Provided swimming assistance"],
                ["Monitored comfort levels", "Adjusted activity intensity"],
                ["Facilitated pool access", "Supported breathing techniques"]
            ],
            "grocery run": [
                ["Assisted with list management", "Provided navigation support"],
                ["Supported item selection", "Managed payment assistance"],
                ["Ensured transport safety", "Organized purchased items"]
            ]
        }

        # Activity-specific contributing factors
        factor_sets = {
            "reading session": [
                ["Lighting conditions", "Text complexity"],
                ["Attention span", "Visual acuity"],
                ["Interest level", "Comprehension ability"]
            ],
            "shopping trip": [
                ["Store crowding", "Budget constraints"],
                ["Decision fatigue", "Mobility challenges"],
                ["Time pressures", "Product availability"]
            ],
            "community art class": [
                ["Fine motor skills", "Creative confidence"],
                ["Peer interaction", "Material familiarity"],
                ["Instruction clarity", "Environmental distractions"]
            ],
            "physio appointment": [
                ["Pain levels", "Exercise tolerance"],
                ["Balance stability", "Motivation levels"],
                ["Equipment adaptation", "Physical limitations"]
            ],
            "swimming": [
                ["Water temperature", "Comfort level"],
                ["Swimming ability", "Safety awareness"],
                ["Pool conditions", "Physical stamina"]
            ],
            "grocery run": [
                ["Store layout", "Memory aids"],
                ["Physical mobility", "Budget awareness"],
                ["Time management", "Decision making"]
            ]
        }

        # Select appropriate actions and factors
        actions = random.choice(action_sets.get(activity, [["Provided support", "Monitored situation"]]))
        factors = random.choice(factor_sets.get(activity, [["Environmental conditions", "Individual needs"]]))

        mock_response = {
            "narrative": narrative,
            "start_time": random.choice(["09:30", "10:15", "11:00", "13:45", "14:30", "15:15", "16:00"]),
            "duration_minutes": random.randint(120, 180),
            "participation": random.choice(participation_options),
            "actions_taken": actions,
            "contributing_factors": factors,
            "productivity_level": random.randint(1, 5),
            "engagement_level": random.randint(1, 3),
            "activity": activity,
            "location": location
        }

        return [{"generated_text": json.dumps(mock_response)}]

    return mock_generate


def get_prompt_template(model_id, activity, location):
    """Get model-specific prompt that encourages narrative generation"""

    for model_key in MODEL_PROMPTS:
        if model_key in model_id:
            return MODEL_PROMPTS[model_key].format(activity=activity, location=location)

    return MODEL_PROMPTS["default"].format(activity=activity, location=location)


def extract_json_with_narrative_validation(text, activity, location):
    """Extract JSON and validate that narrative is properly generated"""

    # Find JSON patterns
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\{.*?\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json_text = match.strip()
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)

                obj = json.loads(json_text)

                # Validate that narrative exists and isn't templated
                if "narrative" in obj:
                    narrative = obj["narrative"]

                    # Check if narrative is too generic/templated
                    generic_phrases = [
                        "Client participated in",
                        "Actions taken included",
                        "Contributing factors included"
                    ]

                    # If narrative contains too many generic phrases, it might be templated
                    generic_count = sum(1 for phrase in generic_phrases if phrase in narrative)

                    if generic_count <= 1:  # Allow some generic language but not all
                        obj["activity"] = activity
                        obj["location"] = location
                        return obj

            except json.JSONDecodeError:
                continue

    # If no good JSON found, return None to trigger regeneration
    return None


@app.command()
def generate(
        activity: str = "reading session",
        location: str = "public library",
        max_attempts: int = 3
) -> dict:
    """Generate incident JSON with dynamic narrative"""

    MODEL_ID = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    generator = load_model()

    for attempt in range(max_attempts):
        try:
            prompt = get_prompt_template(MODEL_ID, activity, location)

            print(f"[blue]Attempt {attempt + 1}: Generating narrative for {activity} at {location}[/]")

            outputs = generator(prompt, return_full_text=False, num_return_sequences=1)
            raw_output = outputs[0]["generated_text"].strip()

            if attempt == 0:  # Show output for first attempt
                print(f"[blue]Raw output preview:[/] {raw_output[:200]}...")

            # Extract and validate JSON
            result = extract_json_with_narrative_validation(raw_output, activity, location)

            if result is not None:
                # Final narrative validation and adjustment
                narrative = result["narrative"]

                # Ensure proper length
                if len(narrative) < 120:
                    # If too short, we might need to regenerate rather than pad
                    if attempt < max_attempts - 1:
                        print(f"[yellow]Narrative too short ({len(narrative)} chars), retrying...[/]")
                        continue
                    else:
                        # Last attempt, pad minimally
                        result["narrative"] = narrative + " Additional support was provided as needed."

                if len(result["narrative"]) > 180:
                    result["narrative"] = result["narrative"][:177] + "..."

                print(f"[green]✓[/] Generated dynamic narrative ({len(result['narrative'])} chars)")
                return result

            else:
                print(f"[yellow]Attempt {attempt + 1} produced templated narrative, retrying...[/]")

        except Exception as e:
            print(f"[red]Attempt {attempt + 1} failed: {e}[/]")

            if attempt == max_attempts - 1:
                print("[yellow]All attempts failed, using enhanced mock generator[/]")
                mock_gen = create_narrative_mock_generator()
                mock_result = mock_gen("", activity=activity, location=location)
                return json.loads(mock_result[0]["generated_text"])

    # Should not reach here, but just in case
    return {
        "narrative": f"Incident occurred during {activity} at {location}. Appropriate support measures were implemented throughout. Various environmental and personal factors influenced the session outcome.",
        "start_time": "14:30",
        "duration_minutes": 150,
        "participation": "participated",
        "actions_taken": ["Provided support", "Monitored situation"],
        "contributing_factors": ["Environmental factors", "Personal factors"],
        "productivity_level": 3,
        "engagement_level": 2,
        "activity": activity,
        "location": location
    }


if __name__ == "__main__":
    app()
'''
#!/usr/bin/env python
import os
import json
import re
import typer
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich import print
from pathlib import Path

load_dotenv()
app = typer.Typer(add_completion=False)

# 选择一个指令式大模型（示例：Falcon 7B Instruct）
MODEL_ID = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")  # 可替换为任何 text-generation 模型 :contentReference[oaicite:2]{index=2}

# 加载 tokenizer 与模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",        # 自动分配 GPU/CPU
    offload_folder="offload", # 大模型时可选
    torch_dtype="auto",
)

# 文本生成 pipeline，自动处理预/后处理
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    do_sample=True,
    temperature=0.7,
)

# 读取 Prompt 模板
TEMPLATE = Path(__file__).with_name("prompt_template.md").read_text(encoding="utf-8")

@app.command()
def generate(
    activity: str = "reading session",
    location: str = "public library"
):
    """生成一条 incident JSON（仅输出 JSON）"""
    prompt = TEMPLATE.format(activity=activity, location=location)
    outputs = generator(prompt, return_full_text=False, num_return_sequences=1)
    # pipeline 返回列表，每项为 {"generated_text": "..."}
    raw = outputs[0]["generated_text"].strip()

    # 确保输出为纯 JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        print("[bold red]✗ 无法在模型输出中找到 JSON 对象，请检查 Prompt 或模型类型[/]")
        print(raw)
        raise typer.Exit(code=1)

    json_text = m.group(0)
    try:
        obj = json.loads(json_text)
        return obj
    except json.JSONDecodeError as e:
        print(f"[bold red]✗ JSON 解析失败：{e}[/]")
        print(json_text)
        raise typer.Exit(code=1)
    '''

'''
    try:
        obj = json.loads(raw)
        print(obj)
        return obj
    except json.JSONDecodeError:
        print("[bold red]✗ 解析失败，请检查模型输出是否为有效 JSON[/]")
        print(raw)
        raise typer.Exit(code=1)
'''


