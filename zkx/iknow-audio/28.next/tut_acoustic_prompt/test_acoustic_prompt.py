# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from tut28_common import run_tut_experiment

OUTPUT_DIR = "/data/zkx/zkx/iknow-audio/28.next/tut_acoustic_prompt"
PROMPT_JSON = os.path.join(OUTPUT_DIR, "llm_prompts_acoustic_qwen.json")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results_acoustic_prompt.json")
MODEL_ROOT = "/data/zkx/zkx/iknow-audio/data/model"
MAX_WORDS = 14
SYSTEM_PROMPT = """You rewrite an acoustic-scene knowledge triple into one short English sentence for audio-text matching.
Rules:
- exactly one sentence
- 6 to 14 words
- must mention the head and tail explicitly
- emphasize audible scene cues, not encyclopedia definitions
- no explanation, no graph wording, no extra clauses
- output only the sentence"""


def normalize_text(text):
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"' ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip() if parts else text


def fallback_prompt(head, rel, tail):
    if rel == "scene contains":
        return f"{head} scene with {tail} sound."
    if rel == "event composed of":
        return f"{head} ambience with {tail} audio."
    if rel == "described by":
        return f"{head} sound scene described by {tail}."
    if rel == "has parent":
        return f"{head} belongs to the {tail} scene."
    if rel == "is variant of":
        return f"{head} is a variant of {tail}."
    return f"{head}, {tail}"


def prompt_is_valid(text, head, tail):
    lower = text.lower()
    if any(token in lower for token in ["graph", "triple", "relation", "knowledge", "therefore", "however"]):
        return False
    if head.lower() not in lower or tail.lower() not in lower:
        return False
    wc = len(text.split())
    if wc < 4 or wc > MAX_WORDS:
        return False
    return True


def choose_qwen_path():
    for name in os.listdir(MODEL_ROOT):
        full = os.path.join(MODEL_ROOT, name)
        if os.path.isdir(full) and name != "gpt2":
            return full
    raise FileNotFoundError(f"No local causal LLM found under {MODEL_ROOT}")


def build_prompt_bank(base, original_prompt_map):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if os.path.exists(PROMPT_JSON):
        return json.loads(Path(PROMPT_JSON).read_text(encoding="utf-8"))

    qwen_path = choose_qwen_path()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(qwen_path, local_files_only=True, torch_dtype=base.torch.float16).to(base.DEVICE)
    model.eval()

    prompt_bank = {}
    keys = list(original_prompt_map.keys())
    for key in base.tqdm(keys, desc="Build TUT acoustic prompt bank"):
        head, rel, tail = [x.strip() for x in key.split("||")]
        prompt = SYSTEM_PROMPT + f"\nRewrite: [{head}, {rel}, {tail}]\nAnswer:"
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        response = normalize_text(response)
        if not prompt_is_valid(response, head, tail):
            response = fallback_prompt(head, rel, tail)
        prompt_bank[key] = response

    Path(PROMPT_JSON).write_text(json.dumps(prompt_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt_bank


def setup_hook(base, prompt_map):
    return {"acoustic_prompt_map": build_prompt_bank(base, prompt_map)}


def experiment_runner(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, context):
    return base.run_selective2hop_originalagg(
        clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes,
        class_labels_set, hop1_relations, hop2_relations, get_tails, context["acoustic_prompt_map"]
    )


if __name__ == "__main__":
    run_tut_experiment(
        exp_title="Starting TUT acoustic-aware prompt bank experiment",
        exp_label="AcousticPrompt",
        output_json=OUTPUT_JSON,
        experiment_runner=experiment_runner,
        extra_meta={"prompt_json": PROMPT_JSON, "model_root": MODEL_ROOT},
        setup_hook=setup_hook,
    )
