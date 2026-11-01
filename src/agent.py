"""Baseline LLM agent for Step 1 — uses Groq API with no document grounding."""

import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Setup Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# High intelligence, excellent multilingual (Hindi/Hinglish/English) reasoning
MODEL_NAME = "openai/gpt-oss-120b"


def create_agent(system_prompt: str):
    """Return an agent configuration bundle."""
    return {
        "model": MODEL_NAME,
        "system_prompt": system_prompt,
    }


def ask(agent_cfg: dict, question: str, max_retries: int = 3) -> str:
    """Send a question to the Groq LLM with retries for rate limits."""
    messages = [
        {"role": "system", "content": agent_cfg["system_prompt"]},
        {"role": "user", "content": question},
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=agent_cfg["model"],
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            err_str = str(e).lower()
            if any(s in err_str for s in ["429", "rate_limit", "quota", "503"]):
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"  ⏳ Rate limited ({attempt + 1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Error: {e}")
                if attempt == max_retries - 1:
                    return f"[ERROR] {e}"
                time.sleep(2)
    return "[ERROR] Max retries exceeded"
