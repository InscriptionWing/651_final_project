#!/usr/bin/env python
# !/usr/bin/env python
"""
Enhanced LLM generation module optimized for RTX 4070 laptop.
Supports both Mistral-7B and smaller models for faster generation.
"""
import os
import json
import re
import typer
import random
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich import print
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import gc

# Model configurations for different speed/quality trade-offs
MODEL_CONFIGS = {
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "torch_dtype": torch.float16,
        "max_new_tokens": 300,  # Increased for complete JSON
        "temperature": 0.7,
        "load_in_8bit": False,  # RTX 4070 can handle fp16
    },
    "mistral-7b-fast": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "torch_dtype": torch.float16,
        "max_new_tokens": 250,  # Increased for complete JSON
        "temperature": 0.8,  # Slightly higher for variety
        "load_in_8bit": True,  # 8-bit for speed and VRAM efficiency
    },
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "torch_dtype": torch.float16,
        "max_new_tokens": 300,  # Increased for complete JSON
        "temperature": 0.7,
        "load_in_8bit": False,
    },
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "torch_dtype": torch.float16,
        "max_new_tokens": 300,  # Increased for complete JSON
        "temperature": 0.7,
        "load_in_8bit": False,  # Small enough to fit in VRAM
    }
}

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
}}[/INST]""",

    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": """<|system|>
You are a professional support worker writing incident reports.</s>
<|user|>
Create a realistic incident scenario during {activity} at {location}. Generate a JSON object with:
- narrative: 120-180 chars describing what happened, actions taken, and contributing factors
- start_time: HH:MM between 09:00-17:00
- duration_minutes: 120-180
- participation: participated/refused/complained/empty
- actions_taken: 2 specific actions
- contributing_factors: 2 specific factors
- productivity_level: 1-5
- engagement_level: 1-3
- activity and location</s>
<|assistant|>""",

    "microsoft/phi-2": """Instruct: Create a realistic support worker incident report for {activity} at {location}.
Output: JSON with narrative (120-180 chars), start_time, duration_minutes (120-180), participation, actions_taken (2 items), contributing_factors (2 items), productivity_level (1-5), engagement_level (1-3).
Generate:""",

    "default": """You are a professional support worker writing an incident report. Create a realistic incident scenario during {activity} at {location}.

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
}}"""
}


def find_and_load_env():
    """Find and load .env file from current, parent, or root directories."""
    current_path = Path(__file__).parent
    for env_path in [
        current_path / '.env',
        current_path.parent / '.env',
        current_path.parent.parent / '.env'
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[green]✓[/] Loaded .env from: {env_path}")
            return True
    print(f"[yellow]⚠[/] No .env file found")
    return False


class OptimizedLLMGenerator:
    """Optimized LLM generator for RTX 4070"""

    def __init__(self, model_config: str = "mistral-7b-fast"):
        self.config = MODEL_CONFIGS.get(model_config, MODEL_CONFIGS["mistral-7b-fast"])
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.model_id = self.config["model_id"]

        # RTX 4070 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def load(self):
        """Load model with RTX 4070 optimizations"""
        if self.generator is not None:
            return self.generator

        HF_TOKEN = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
        print(f"[blue]Loading {self.model_id} optimized for RTX 4070...[/]")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=HF_TOKEN
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "token": HF_TOKEN,
                "torch_dtype": self.config["torch_dtype"],
                "low_cpu_mem_usage": True,
            }

            # Load with 8-bit quantization if specified
            if self.config.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
                # Enable CPU offload for limited VRAM
                model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
                # Use auto device map but allow CPU offload
                model_kwargs["device_map"] = "auto"
                # Set max memory to prevent OOM
                model_kwargs["max_memory"] = {0: "5GB", "cpu": "20GB"}
            else:
                # For fp16, use more conservative memory settings
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = {0: "6GB", "cpu": "20GB"}

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Create pipeline with optimizations
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=True,
                temperature=self.config["temperature"],
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                torch_dtype=self.config["torch_dtype"],
            )

            print(f"[green]✓[/] Successfully loaded {self.model_id}")

            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()

            return self.generator

        except Exception as e:
            print(f"[red]✗[/] Failed to load {self.model_id}: {e}")
            
            # Try fallback to smaller model if main model fails
            if "mistral" in self.model_id.lower():
                print(f"[yellow]⚠[/] Trying fallback to TinyLlama for lower VRAM usage...")
                try:
                    self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    self.config = MODEL_CONFIGS["tinyllama"]
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        token=HF_TOKEN
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Load smaller model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                        torch_dtype=self.config["torch_dtype"],
                        device_map="auto",
                        max_memory={0: "4GB", "cpu": "16GB"}
                    )
                    
                    # Create pipeline
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=True,
                        temperature=self.config["temperature"],
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        torch_dtype=self.config["torch_dtype"],
                    )
                    
                    print(f"[green]✓[/] Successfully loaded fallback model: {self.model_id}")
                    return self.generator
                    
                except Exception as fallback_e:
                    print(f"[red]✗[/] Fallback model also failed: {fallback_e}")
            
            raise

    def generate(self, activity: str, location: str, batch_size: int = 1) -> List[dict]:
        """Generate with batching support"""
        if self.generator is None:
            self.load()

        # Get appropriate prompt template
        prompt_template = MODEL_PROMPTS.get(self.model_id, MODEL_PROMPTS["default"])

        # Generate prompts for batch
        prompts = []
        for _ in range(batch_size):
            prompt = prompt_template.format(activity=activity, location=location)
            prompts.append(prompt)

        try:
            # Generate in batch
            outputs = self.generator(
                prompts,
                return_full_text=False,
                num_return_sequences=1,
                batch_size=batch_size
            )
            
            print(f"[debug] Pipeline outputs: {len(outputs)} items")
            if outputs and len(outputs) > 0:
                print(f"[debug] First output structure: {type(outputs[0])}")
                if hasattr(outputs[0], 'keys'):
                    print(f"[debug] First output keys: {list(outputs[0].keys())}")

        except Exception as e:
            print(f"[red]Generation failed: {e}[/]")
            # Fallback to single generation
            outputs = []
            for prompt in prompts:
                try:
                    single_output = self.generator(
                        prompt,
                        return_full_text=False,
                        num_return_sequences=1
                    )
                    outputs.extend(single_output)
                except Exception as single_e:
                    print(f"[red]Single generation failed: {single_e}[/]")

        results = []
        for output in outputs:
            try:
                # Handle different output formats
                if isinstance(output, dict):
                    if "generated_text" in output:
                        raw_text = output["generated_text"].strip()
                    else:
                        print(f"[debug] Output dict keys: {list(output.keys())}")
                        continue
                elif isinstance(output, list) and len(output) > 0:
                    if isinstance(output[0], dict) and "generated_text" in output[0]:
                        raw_text = output[0]["generated_text"].strip()
                    else:
                        print(f"[debug] Output list[0] type: {type(output[0])}")
                        continue
                else:
                    print(f"[debug] Unexpected output type: {type(output)}")
                    continue
                
                print(f"[debug] Raw text: {raw_text[:100]}...")
                json_obj = self._extract_json(raw_text, activity, location)
                if json_obj:
                    results.append(json_obj)
                    print(f"[debug] Successfully extracted JSON")
                else:
                    print(f"[debug] Failed to extract JSON from: {raw_text[:100]}...")
                    
            except Exception as e:
                print(f"[red]Error processing output: {e}[/]")
                continue

        print(f"[debug] Generated {len(results)} valid results from {len(outputs)} outputs")
        return results

    def _extract_json(self, text: str, activity: str, location: str) -> Optional[dict]:
        """Extract and validate JSON from LLM output"""
        print(f"[debug] Extracting JSON from text: {text[:200]}...")
        
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
                    
                    print(f"[debug] Trying to parse JSON: {json_text[:100]}...")

                    obj = json.loads(json_text)
                    obj["activity"] = activity
                    obj["location"] = location

                    # Basic validation
                    if "narrative" in obj and len(obj["narrative"]) >= 50:
                        print(f"[debug] Successfully parsed JSON with narrative: {obj['narrative'][:50]}...")
                        return obj
                    else:
                        print(f"[debug] JSON parsed but missing narrative or too short")

                except json.JSONDecodeError as e:
                    print(f"[debug] JSON decode error: {e}")
                    # Try to fix common JSON issues
                    try:
                        # Try to complete incomplete JSON
                        if json_text.count('{') > json_text.count('}'):
                            # Missing closing braces
                            missing_braces = json_text.count('{') - json_text.count('}')
                            json_text += '}' * missing_braces
                            
                        # Try to add missing quotes around keys
                        json_text = re.sub(r'(\w+):', r'"\1":', json_text)
                        
                        obj = json.loads(json_text)
                        obj["activity"] = activity
                        obj["location"] = location
                        
                        if "narrative" in obj and len(obj["narrative"]) >= 50:
                            print(f"[debug] Fixed JSON successfully")
                            return obj
                            
                    except:
                        continue

        # If no JSON found, try to create a basic structure from the text
        print(f"[debug] No valid JSON found, attempting to create basic structure")
        try:
            # Extract narrative from text
            narrative_match = re.search(r'"narrative":\s*"([^"]+)"', text)
            if narrative_match:
                narrative = narrative_match.group(1)
                if len(narrative) >= 50:
                    return {
                        "narrative": narrative,
                        "start_time": "14:30",
                        "duration_minutes": 150,
                        "participation": "participated",
                        "actions_taken": ["Provided support", "Monitored situation"],
                        "contributing_factors": ["Environmental factors", "Individual needs"],
                        "productivity_level": 3,
                        "engagement_level": 2,
                        "activity": activity,
                        "location": location
                    }
        except:
            pass

        return None


# Global generator instance
_generator_instance = None
_generator_config = None


def get_generator(config: str = "mistral-7b-fast") -> OptimizedLLMGenerator:
    """Get or create generator instance"""
    global _generator_instance, _generator_config

    if _generator_instance is None or _generator_config != config:
        _generator_instance = OptimizedLLMGenerator(config)
        _generator_config = config
        _generator_instance.load()

    return _generator_instance


# Load environment variables
find_and_load_env()
app = typer.Typer(add_completion=False)


@app.command()
def generate(
        activity: str = "reading session",
        location: str = "public library",
        max_attempts: int = 2,
        model_config: str = "mistral-7b-fast"
) -> dict:
    """Generate incident JSON using optimized LLM"""

    generator = get_generator(model_config)

    for attempt in range(max_attempts):
        try:
            print(f"[blue]Generating for {activity} at {location} (attempt {attempt + 1})...[/]")

            results = generator.generate(activity, location, batch_size=1)

            if results and len(results) > 0:
                result = results[0]

                # Ensure narrative length
                narrative = result.get("narrative", "")
                if len(narrative) < 120:
                    result["narrative"] = narrative + " Additional support was provided as needed."
                if len(result["narrative"]) > 180:
                    result["narrative"] = result["narrative"][:177] + "..."

                print(f"[green]✓[/] Generated narrative ({len(result['narrative'])} chars)")
                return result

        except Exception as e:
            print(f"[red]Attempt {attempt + 1} failed: {e}[/]")

            if attempt == max_attempts - 1:
                # Fallback response
                return {
                    "narrative": f"Client participated in {activity} at {location}. Support provided throughout with actions taken to address contributing factors. Session went as planned.",
                    "start_time": "14:30",
                    "duration_minutes": 150,
                    "participation": "participated",
                    "actions_taken": ["Provided support", "Monitored situation"],
                    "contributing_factors": ["Environmental factors", "Individual needs"],
                    "productivity_level": 3,
                    "engagement_level": 2,
                    "activity": activity,
                    "location": location
                }

    raise Exception("Failed to generate after all attempts")


@app.command()
def benchmark(count: int = 10):
    """Benchmark different model configurations"""
    import time

    configs = ["tinyllama", "phi-2", "mistral-7b-fast", "mistral-7b"]
    activities = ["reading session", "shopping trip", "swimming"]
    locations = ["public library", "community centre", "local pool"]

    for config in configs:
        try:
            print(f"\n[bold cyan]Testing {config}...[/]")
            start = time.time()

            generator = get_generator(config)
            successes = 0

            for i in range(count):
                activity = random.choice(activities)
                location = random.choice(locations)

                try:
                    result = generate(activity, location, max_attempts=1, model_config=config)
                    if result and "narrative" in result:
                        successes += 1
                except:
                    pass

            elapsed = time.time() - start
            rate = (successes / elapsed) * 3600 if elapsed > 0 else 0

            print(f"[green]✓[/] {config}: {successes}/{count} in {elapsed:.1f}s")
            print(f"    Rate: {rate:.0f} records/hour")

            # Clear memory between tests
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[red]✗[/] {config} failed: {e}")


@app.command()
def batch_generate(
        count: int = 100,
        batch_size: int = 4,
        model_config: str = "mistral-7b-fast"
) -> List[dict]:
    """Generate multiple records using batching"""
    import time

    activities = ["reading session", "shopping trip", "community art class",
                  "physio appointment", "swimming", "grocery run"]
    locations = ["public library", "Westfield Mall", "community centre",
                 "local pool", "physiotherapy clinic"]

    generator = get_generator(model_config)
    results = []

    print(f"[blue]Generating {count} records in batches of {batch_size}...[/]")
    start_time = time.time()

    for i in range(0, count, batch_size):
        batch_count = min(batch_size, count - i)
        activity = random.choice(activities)
        location = random.choice(locations)

        try:
            batch_results = generator.generate(activity, location, batch_size=batch_count)
            results.extend(batch_results)

            if (i + batch_count) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (len(results) / elapsed) * 3600
                print(f"[dim]Progress: {len(results)}/{count} - Rate: {rate:.0f}/hour[/]")

        except Exception as e:
            print(f"[yellow]Batch {i // batch_size + 1} failed: {e}[/]")

    elapsed = time.time() - start_time
    print(f"\n[green]✓[/] Generated {len(results)} records in {elapsed:.1f}s")
    print(f"    Rate: {(len(results) / elapsed) * 3600:.0f} records/hour")

    return results


# ver 5
"""
Enhanced LLM generation module with external template support.
Supports both real model inference and mock generation with external templates.
"""
# Note: Ensure we use PyTorch's torch, not safetensors.torch to avoid shadowing backends
'''
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
}}[/INST]""",

    "default": """You are a professional support worker writing an incident report. Create a realistic incident scenario during {activity} at {location}.

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
}}"""
}

# run_llm.py
#!/usr/bin/env python
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 1. 只加载一次模型
load_dotenv(override=True)
MODEL_ID  = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HUB_TOKEN = os.getenv("HF_HUB_TOKEN") or None

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, token=HUB_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    token=HUB_TOKEN
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
)

# 2. 只读一次模板
TEMPLATE = Path(__file__).with_name("prompt_template.md").read_text(encoding="utf-8")

def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("No JSON in LLM output")
    return json.loads(m.group(0))

def generate(activity: str, location: str) -> dict:
    """
    兼容单条生成
    """
    prompt = TEMPLATE.replace("{activity}", activity).replace("{location}", location)
    out = generator(prompt, return_full_text=False, num_return_sequences=1)[0]["generated_text"]
    return _extract_json(out)

def generate_batch(
    activities: list[str],
    locations:  list[str]
) -> list[dict]:
    """
    批量生成接口
    """
    prompts = [
        TEMPLATE.replace("{activity}", a).replace("{location}", l)
        for a, l in zip(activities, locations)
    ]
    outputs = generator(prompts, return_full_text=False)
    results = []
    for o in outputs:
        try:
            results.append(_extract_json(o["generated_text"]))
        except Exception:
            continue
    return results

'''


# ver 4
'''
import os
import json
import re
import typer
import random
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich import print
from pathlib import Path
from typing import Dict, List, Any, Optional


def find_and_load_env():
    """Find and load .env file from current, parent, or root directories."""
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


class TemplateLoader:
    """Load and manage external templates for incident generation."""

    def __init__(self, templates_dir: str = None):
        """
        Initialize template loader.

        Args:
            templates_dir: Directory containing template files. Defaults to current directory.
        """
        if templates_dir is None:
            self.templates_dir = Path(__file__).parent
        else:
            self.templates_dir = Path(templates_dir)

        self._narrative_templates = None
        self._action_templates = None
        self._factor_templates = None

    def load_narrative_templates(self, filename: str = "narrative_templates.json") -> Dict[str, List[str]]:
        """Load narrative templates from JSON file."""
        if self._narrative_templates is None:
            try:
                template_path = self.templates_dir / filename
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self._narrative_templates = json.load(f)
                    print(f"[green]✓[/] Loaded narrative templates from {template_path}")
                else:
                    print(f"[yellow]⚠[/] Narrative template file not found: {template_path}")
                    self._narrative_templates = self._get_default_narrative_templates()
            except (json.JSONDecodeError, IOError) as e:
                print(f"[red]✗[/] Error loading narrative templates: {e}")
                self._narrative_templates = self._get_default_narrative_templates()

        return self._narrative_templates

    def load_action_templates(self, filename: str = "action_templates.json") -> Dict[str, List[List[str]]]:
        """Load action templates from JSON file."""
        if self._action_templates is None:
            try:
                template_path = self.templates_dir / filename
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self._action_templates = json.load(f)
                    print(f"[green]✓[/] Loaded action templates from {template_path}")
                else:
                    print(f"[yellow]⚠[/] Action template file not found: {template_path}")
                    self._action_templates = self._get_default_action_templates()
            except (json.JSONDecodeError, IOError) as e:
                print(f"[red]✗[/] Error loading action templates: {e}")
                self._action_templates = self._get_default_action_templates()

        return self._action_templates

    def load_factor_templates(self, filename: str = "factor_templates.json") -> Dict[str, List[List[str]]]:
        """Load factor templates from JSON file."""
        if self._factor_templates is None:
            try:
                template_path = self.templates_dir / filename
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self._factor_templates = json.load(f)
                    print(f"[green]✓[/] Loaded factor templates from {template_path}")
                else:
                    print(f"[yellow]⚠[/] Factor template file not found: {template_path}")
                    self._factor_templates = self._get_default_factor_templates()
            except (json.JSONDecodeError, IOError) as e:
                print(f"[red]✗[/] Error loading factor templates: {e}")
                self._factor_templates = self._get_default_factor_templates()

        return self._factor_templates

    def load_all_templates(self) -> tuple:
        """Load all templates and return as tuple."""
        narratives = self.load_narrative_templates()
        actions = self.load_action_templates()
        factors = self.load_factor_templates()
        return narratives, actions, factors

    def reload_templates(self):
        """Force reload all templates from files."""
        self._narrative_templates = None
        self._action_templates = None
        self._factor_templates = None

    def get_supported_activities(self) -> List[str]:
        """Get list of activities that have templates."""
        narratives = self.load_narrative_templates()
        return list(narratives.keys())

    @staticmethod
    def _get_default_narrative_templates() -> Dict[str, List[str]]:
        """Fallback narrative templates if file loading fails."""
        return {
            "reading session": [
                "Client showed interest in adventure novels but became frustrated with small text. Provided magnifying glass and encouraged shorter sessions. Client's vision difficulties and lighting conditions contributed to challenges.",
                "During reading session, client selected cookbook and engaged well initially. Became restless after 45 minutes. Offered break and different seating. Time of day and attention span were key factors.",
                "Client participated in group reading activity. Initially shy but warmed up when discussing favorite characters. Facilitated introductions and provided encouragement. Social anxiety and group dynamics influenced participation.",
                "Client chose mystery novel and read quietly for extended period. Asked several questions about plot. Provided explanations and praise for engagement. Client's curiosity and genre preference enhanced experience."
            ],
            "shopping trip": [
                "Client managed shopping list well but became overwhelmed in crowded aisles. Guided to quieter sections and helped prioritize items. Store crowds and sensory overload were contributing factors.",
                "During grocery shopping, client forgot wallet and became distressed. Reassured client and contacted family for support. Memory challenges and anxiety affected the outing experience.",
                "Client enjoyed selecting fresh produce and comparing prices. Needed assistance reading small labels. Provided reading support and praised decision-making skills. Visual impairment and independence goals were factors.",
                "Shopping trip went smoothly until checkout queues caused frustration. Found shorter line and used calming techniques. Wait times and impatience were main contributing elements."
            ],
            "community art class": [
                "Client enthusiastically started watercolor painting but struggled with brush control. Demonstrated techniques and provided adaptive brushes. Fine motor skills and equipment suitability affected performance.",
                "During art session, client felt intimidated by others' work. Encouraged personal expression and highlighted unique style. Comparison anxiety and self-confidence were significant factors.",
                "Client created beautiful collage but spilled paint on clothing. Helped clean up and reassured about accidents. Coordination challenges and material management influenced the session.",
                "Art class progressed well with client showing creativity in sculpture. Needed support with tool usage. Provided guidance and celebrated achievements. Motor skills and artistic confidence were key."
            ],
            "physio appointment": [
                "Client arrived early and showed motivation for exercises. Complained of soreness midway through session. Adjusted intensity and provided rest breaks. Pain levels and exercise tolerance were factors.",
                "During physiotherapy, client struggled with balance exercises. Provided additional support and modified routine. Inner ear issues and fear of falling contributed to difficulties.",
                "Client completed prescribed exercises but rushed through repetitions. Encouraged slower pace and proper form. Impatience and understanding of technique importance were influences.",
                "Physiotherapy session successful with client achieving new range of motion goals. Celebrated progress and planned next steps. Consistent attendance and effort were contributing positives."
            ],
            "swimming": [
                "Client enjoyed pool time but hesitated at deep end. Stayed in shallow area and provided reassurance. Previous negative experience and water depth anxiety were key factors.",
                "During swimming session, client showed improved stroke technique. Needed breaks every few lengths. Provided encouragement and paced activity. Fitness level and breathing control influenced performance.",
                "Client participated in water aerobics class with enthusiasm. Required assistance entering pool. Helped with mobility and safety precautions. Physical limitations and pool access were considerations.",
                "Swimming went well until other pool users became noisy. Moved to quieter lane and continued exercises. Sensory sensitivity and environmental distractions affected concentration."
            ],
            "grocery run": [
                "Client managed small shopping list independently but forgot milk. Returned to dairy section together and completed task. Memory aids and routine establishment were important factors.",
                "During grocery run, client became confused by store layout changes. Provided navigation help and found familiar products. Store modifications and adaptation challenges influenced the trip.",
                "Client selected healthy options and stayed within budget constraints. Praised good choices and mathematical skills. Nutritional awareness and financial management were positive contributors.",
                "Grocery shopping complicated by client's mobility device in narrow aisles. Found accessible routes and assisted with reaching items. Store design and physical accessibility were factors."
            ]
        }

    @staticmethod
    def _get_default_action_templates() -> Dict[str, List[List[str]]]:
        """Fallback action templates if file loading fails."""
        return {
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

    @staticmethod
    def _get_default_factor_templates() -> Dict[str, List[List[str]]]:
        """Fallback factor templates if file loading fails."""
        return {
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


# Load environment variables
find_and_load_env()

app = typer.Typer(add_completion=False)

# Global variables
tokenizer = None
model = None
generator = None
template_loader = TemplateLoader()


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
            token=HF_TOKEN
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200, # max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=1,  # Add this
            torch_dtype=torch.float16,
        )

        print(f"[green]✓[/] Successfully loaded {MODEL_ID} for narrative generation")
        return generator

    except Exception as e:
        print(f"[red]✗[/] Failed to load {MODEL_ID}: {e}")
        return create_narrative_mock_generator()


def create_narrative_mock_generator():
    """Enhanced mock generator using external templates"""
    print("[yellow]⚠ Using narrative mock generator with external templates[/]")

    # Load templates from external files
    try:
        narrative_templates, action_templates, factor_templates = template_loader.load_all_templates()
        print(f"[green]✓[/] Loaded templates for {len(narrative_templates)} activities")
    except Exception as e:
        print(f"[red]✗[/] Failed to load external templates: {e}")
        # Use minimal fallback
        narrative_templates = template_loader._get_default_narrative_templates()
        action_templates = template_loader._get_default_action_templates()
        factor_templates = template_loader._get_default_factor_templates()

    def mock_generate(prompt, **kwargs):
        # Extract activity and location from prompt
        activity = "reading session"
        location = "public library"

        # Extract from prompt
        for act in narrative_templates.keys():
            if act in prompt.lower():
                activity = act
                break

        locations = ["public library", "westfield mall", "community centre", "local pool", "physiotherapy clinic"]
        for loc in locations:
            if loc in prompt.lower():
                location = loc
                break

        # Get realistic narrative for activity
        if activity in narrative_templates:
            narrative = random.choice(narrative_templates[activity])
        else:
            # Fallback for unknown activities
            narrative = f"Client engaged in {activity} with varying levels of participation. Support provided throughout session with attention to individual needs. Environmental and personal factors influenced the experience."

        # Ensure narrative is within character limit
        if len(narrative) < 120:
            narrative += " Additional observations and support strategies were documented for future reference."
        if len(narrative) > 180:
            narrative = narrative[:177] + "..."

        # Generate realistic supporting data
        participation_options = ["participated", "refused", "complained", ""]

        # Select appropriate actions and factors
        actions = random.choice(action_templates.get(activity, [["Provided support", "Monitored situation"]]))
        factors = random.choice(factor_templates.get(activity, [["Environmental conditions", "Individual needs"]]))

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
        max_attempts: int = 3,
        templates_dir: Optional[str] = None
) -> dict:
    """Generate incident JSON with dynamic narrative using external templates"""

    # Update template loader directory if specified
    if templates_dir:
        global template_loader
        template_loader = TemplateLoader(templates_dir)

    MODEL_ID = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    generator_func = load_model()

    for attempt in range(max_attempts):
        try:
            prompt = get_prompt_template(MODEL_ID, activity, location)

            print(f"[blue]Attempt {attempt + 1}: Generating narrative for {activity} at {location}[/]")

            outputs = generator_func(prompt, return_full_text=False, num_return_sequences=1)
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


@app.command()
def list_supported_activities(templates_dir: Optional[str] = None):
    """List all activities supported by current templates"""
    if templates_dir:
        loader = TemplateLoader(templates_dir)
    else:
        loader = template_loader

    activities = loader.get_supported_activities()

    if activities:
        print(f"[bold cyan]Supported Activities ({len(activities)}):[/]")
        for activity in sorted(activities):
            print(f"  • {activity}")
    else:
        print("[red]No activities found in templates[/]")


@app.command()
def validate_templates(templates_dir: Optional[str] = None):
    """Validate external template files"""
    if templates_dir:
        loader = TemplateLoader(templates_dir)
    else:
        loader = template_loader

    try:
        narratives, actions, factors = loader.load_all_templates()

        print(f"[green]✓[/] Successfully loaded:")
        print(f"  • Narratives: {len(narratives)} activities")
        print(f"  • Actions: {len(actions)} activities")
        print(f"  • Factors: {len(factors)} activities")

        # Check for narrative length issues
        length_issues = []
        for activity, narrative_list in narratives.items():
            for i, narrative in enumerate(narrative_list):
                length = len(narrative)
                if length < 120 or length > 180:
                    length_issues.append(f"{activity} narrative {i + 1}: {length} chars")

        if length_issues:
            print(f"\n[yellow]⚠ Narrative length issues:[/]")
            for issue in length_issues[:10]:  # Show first 10
                print(f"  • {issue}")
            if len(length_issues) > 10:
                print(f"  • ... and {len(length_issues) - 10} more")
        else:
            print(f"\n[green]✓[/] All narratives within length requirements")

    except Exception as e:
        print(f"[red]✗[/] Template validation failed: {e}")
        return False

    return True


@app.command()
def reload_templates():
    """Reload templates from external files"""
    global template_loader
    template_loader.reload_templates()
    print("[green]✓[/] Templates reloaded from external files")


if __name__ == "__main__":
    app()
'''

'''
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


