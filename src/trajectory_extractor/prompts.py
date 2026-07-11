import json
from pathlib import Path


def load_prompts(path: str | Path) -> dict[str, str]:
    prompts_path = Path(path)
    data = json.loads(prompts_path.read_text())

    if not isinstance(data, dict):
        raise ValueError("Prompt file must contain a JSON object mapping names to prompt text.")

    prompts: dict[str, str] = {}
    for condition_name, prompt_text in data.items():
        if not isinstance(condition_name, str) or not condition_name.strip():
            raise ValueError("Prompt condition names must be non-empty strings.")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"Prompt text for {condition_name!r} must be a non-empty string.")
        prompts[condition_name] = prompt_text

    return prompts
