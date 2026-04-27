import importlib.util
import io
import json
import os
import re
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import pyarrow.parquet as pq

BASE_MODULE_PATH = "/data/zkx/zkx/iknow-audio/57.vgg_ablation17_vgg.py"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/zkx/zkx/iknow-audio/58.vgg_llm_clean_ablation17")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results_ablation.json")
PROGRESS_JSON = os.path.join(OUTPUT_DIR, "progress.json")
TMP_AUDIO_DIR = os.path.join(OUTPUT_DIR, "tmp_audio")
PROMPT_JSON = os.path.join(OUTPUT_DIR, "llm_prompts_vgg_clean_qwen.json")
PROMPT_PROGRESS_JSON = os.path.join(OUTPUT_DIR, "prompt_gen_progress.json")
LABEL_MAP_JSON = os.path.join(OUTPUT_DIR, "label_map.json")
SAVE_EVERY_PROMPTS = 20
SAVE_EVERY_SAMPLES = 50
MAX_WORDS = 12
MIN_WORDS = 4
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
FORCE_REBUILD_PROMPTS = os.getenv("FORCE_REBUILD_PROMPTS", "0") == "1"

QWEN_CANDIDATES = [
    "/data/zkx/zkx/iknow-audio/data/model/千问",
    "/data/zkx/zkx/iknow-audio/data/model/千问模型",
]

SYSTEM_PROMPT = """You rewrite an audio knowledge graph triple into one short English prompt for audio-text matching.
Rules:
- Write exactly one short sentence.
- Use 5 to 10 words.
- Keep the sound/event head explicit.
- Mention the tail clue in an audio-natural way.
- No explanation, no extra clause, no graph words.
- Output only the sentence."""

EXAMPLES = """Examples:
[playing violin, has parent, string instrument] -> playing violin is a string instrument sound.
[dog barking, event composed of, bark] -> dog barking includes sharp bark sounds.
[female speech, belongs to class, human vocal sound] -> female speech is a human vocal sound.
[airplane flyby, has parent, aircraft] -> airplane flyby is an aircraft sound."""

BANNED = [
    "however", "therefore", "graph", "triple", "relation", "knowledge", "broader class",
    "category", "instance of", "model answer", "correct format", "based solely"
]


def load_module(path):
    spec = importlib.util.spec_from_file_location("vgg_base_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_MODULE_PATH)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TMP_AUDIO_DIR).mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    return " ".join(str(text).replace("_", " ").strip().split())


EXACT_LABEL_MAP = {
    "playing violin, fiddle": "playing violin",
    "female speech, woman speaking": "female speech",
    "male speech, man speaking": "male speech",
    "child speech, kid speaking": "child speech",
    "subway, metro, underground": "subway",
    "race car, auto racing": "race car",
    "electric shaver, electric razor shaving": "electric shaver",
    "bee, wasp, etc. buzzing": "buzzing insect",
    "alligators, crocodiles hissing": "alligator hissing",
    "donkey, ass braying": "donkey braying",
    "cattle, bovinae cowbell": "cowbell",
    "people marching": "marching",
    "people coughing": "coughing",
    "people sneezing": "sneezing",
    "people slurping": "slurping",
    "people whistling": "whistling",
}


def normalize_vgg_label(text):
    x = normalize_text(text).lower()
    if x in EXACT_LABEL_MAP:
        return EXACT_LABEL_MAP[x]
    x = x.replace("&", "and")
    x = re.sub(r"\betc\.\b", "", x)
    x = re.sub(r"\s+", " ", x).strip(" ,")
    return x


def choose_qwen_path():
    for path in QWEN_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"No local Qwen model found in {QWEN_CANDIDATES}")


def load_tokenizer_and_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qwen_path = choose_qwen_path()
    print(f"Loading local Qwen from: {qwen_path}")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_path, dtype=base.torch.float16, local_files_only=True).cuda()
    return tokenizer, model


def fallback_prompt(head, rel, tail):
    h = head.strip()
    t = tail.strip()
    if rel == "belongs to class":
        return f"{h} is a {t} sound."
    if rel == "has parent":
        return f"{h} is a kind of {t} sound."
    if rel == "event composed of":
        return f"{h} includes {t} sounds."
    return f"{h}, {t}"


def prompt_is_valid(text, head=None, tail=None):
    if not text:
        return False
    lower = text.lower()
    if any(token in lower for token in BANNED):
        return False
    if any(ch in text for ch in ["[", "]", "{", "}", ":", ";"]):
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    wc = len(text.split())
    if wc < MIN_WORDS or wc > MAX_WORDS:
        return False
    tokens = text.split()
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'/-]*[.,!?]?", token) for token in tokens):
        return False
    if head and head.lower() not in lower:
        return False
    if tail and tail.lower() not in lower:
        return False
    return True


def generate_short_prompt(tokenizer, model, head, rel, tail):
    prompt = SYSTEM_PROMPT.strip() + "\n\n" + EXAMPLES.strip() + f"\nNow rewrite: [{head}, {rel}, {tail}]\nAnswer:"
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    response = normalize_text(response).strip("\"' ")
    pieces = re.split(r"(?<=[.!?])\s+", response)
    if pieces:
        response = pieces[0].strip()
    if not prompt_is_valid(response, head=head, tail=tail):
        response = fallback_prompt(head, rel, tail)
    return response


def save_prompt_bank(prompt_bank, progress):
    with open(PROMPT_JSON + ".tmp", "w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)
    os.replace(PROMPT_JSON + ".tmp", PROMPT_JSON)
    with open(PROMPT_PROGRESS_JSON + ".tmp", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(PROMPT_PROGRESS_JSON + ".tmp", PROMPT_PROGRESS_JSON)


def load_dataset():
    root = Path(base.VGG_DIR)
    shard_paths = sorted(root.glob("test-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No test parquet shards found in {base.VGG_DIR}")

    sample_meta = []
    raw_labels = set()
    for shard_idx, shard_path in enumerate(shard_paths):
        table = pq.read_table(shard_path, columns=["caption"])
        captions = [normalize_text(x) for x in table["caption"].to_pylist()]
        for row_idx, raw_cap in enumerate(captions):
            raw_labels.add(raw_cap)
            sample_meta.append({"shard_idx": shard_idx, "row_idx": row_idx, "raw_label": raw_cap})

    raw_to_norm = {raw: normalize_vgg_label(raw) for raw in sorted(raw_labels)}
    if len(set(raw_to_norm.values())) != len(raw_to_norm):
        raise RuntimeError("Normalized VGG labels are not unique; refine the mapping before running.")

    label_classes = sorted(set(raw_to_norm.values()))
    class_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    for item in sample_meta:
        item["label"] = raw_to_norm[item["raw_label"]]
        item["true_indices"] = [class_to_idx[item["label"]]]

    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(raw_to_norm, f, ensure_ascii=False, indent=2)

    return {
        "label_classes": label_classes,
        "kg_classes": label_classes[:],
        "class_labels_set": {name.lower() for name in label_classes},
        "samples": sample_meta[:MAX_SAMPLES] if MAX_SAMPLES > 0 else sample_meta,
        "shard_paths": [str(p) for p in shard_paths],
        "total": min(len(sample_meta), MAX_SAMPLES) if MAX_SAMPLES > 0 else len(sample_meta),
        "raw_to_norm": raw_to_norm,
    }


def build_prompt_source_keys(label_classes, class_labels_set, get_tails_fn):
    keys = []
    key_set = set()
    hop1_relations = ["has parent", "event composed of", "belongs to class"]
    hop2_relations = ["has parent", "event composed of"]
    for class_name in label_classes:
        kg_ent = normalize_vgg_label(class_name)
        h1_map = OrderedDict()
        head_norm = kg_ent.lower()
        for rel in hop1_relations:
            for tail in get_tails_fn(kg_ent, rel):
                tail_norm = normalize_text(tail).lower()
                if tail_norm == class_name.lower() or tail_norm in class_labels_set:
                    continue
                key = f"{head_norm}||{rel.lower()}||{tail_norm}"
                if key not in key_set:
                    key_set.add(key)
                    keys.append((key, class_name, rel, normalize_text(tail)))
                if tail_norm not in h1_map:
                    h1_map[tail_norm] = tail
        for h1_tail_norm in h1_map.keys():
            for rel in hop2_relations:
                for tail in get_tails_fn(h1_tail_norm, rel):
                    tail_norm = normalize_text(tail).lower()
                    if tail_norm == class_name.lower() or tail_norm in class_labels_set or tail_norm in h1_map:
                        continue
                    key = f"{h1_tail_norm}||{rel.lower()}||{tail_norm}"
                    if key not in key_set:
                        key_set.add(key)
                        keys.append((key, class_name, rel, normalize_text(tail)))
    return keys


def build_vgg_prompt_bank(label_classes, class_labels_set, get_tails_fn):
    if os.path.exists(PROMPT_JSON) and not FORCE_REBUILD_PROMPTS:
        with open(PROMPT_JSON, "r", encoding="utf-8") as f:
            bank = json.load(f)
        return bank

    keys = build_prompt_source_keys(label_classes, class_labels_set, get_tails_fn)
    tokenizer, model = load_tokenizer_and_model()
    prompt_bank = {}
    fallback_count = 0
    kept_valid_count = 0
    print(f"Building VGG clean prompt bank across {len(keys)} entries")
    for idx in base.tqdm(range(len(keys)), desc="VGG clean prompt generation"):
        key, head, rel, tail = keys[idx]
        generated = generate_short_prompt(tokenizer, model, head, rel, tail)
        if generated == fallback_prompt(head, rel, tail):
            fallback_count += 1
        else:
            kept_valid_count += 1
        prompt_bank[key] = generated
        if (idx + 1) % SAVE_EVERY_PROMPTS == 0:
            save_prompt_bank(prompt_bank, {
                "total_keys": len(keys),
                "completed_keys": idx + 1,
                "fallback_count": fallback_count,
                "kept_valid_count": kept_valid_count,
            })
    save_prompt_bank(prompt_bank, {
        "total_keys": len(keys),
        "completed_keys": len(keys),
        "fallback_count": fallback_count,
        "kept_valid_count": kept_valid_count,
    })
    return prompt_bank


@base.torch.no_grad()
def main():
    print("Starting VGG clean+LLM ablation: simplified labels + event-safe relations + Qwen prompts")
    print(f"Device: {base.DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_AUDIO_DIR, exist_ok=True)

    clap_model = base.CLAP(version="2023", use_cuda=base.torch.cuda.is_available())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kge_model = base.torch.load(os.path.join(base.KGE_MODEL_DIR, "trained_model.pkl"), map_location=base.DEVICE)
    kge_model.eval()
    training_factory = base.TriplesFactory.from_path(base.TRAIN_TRIPLES_PATH)

    hop1_relations = [rel for rel in ["has parent", "event composed of", "belongs to class"] if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in ["has parent", "event composed of"] if rel in training_factory.relation_to_id]
    print(f"Valid hop1 relations: {hop1_relations}")
    print(f"Valid hop2 relations: {hop2_relations}")
    get_tails = base.build_tail_predictor(kge_model, training_factory)

    dataset = load_dataset()
    label_classes = dataset["label_classes"]
    kg_classes = dataset["kg_classes"]
    class_labels_set = dataset["class_labels_set"]
    samples = dataset["samples"]
    shard_paths = dataset["shard_paths"]
    total_samples = len(samples)
    print(f"Total samples prepared: {total_samples}")
    print(f"Unique normalized labels: {len(label_classes)}")

    prompt_map = build_vgg_prompt_bank(label_classes, class_labels_set, get_tails)
    text_embeds = base.F.normalize(base.get_safe_text_embeddings(clap_model, label_classes, base.DEVICE), dim=-1)

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                done_payload = json.load(f)
            if done_payload.get("completed") and int(done_payload.get("total_samples", -1)) == total_samples:
                print(f"Found completed results: {OUTPUT_JSON}")
                return
        except Exception:
            pass

    progress = base.load_progress(total_samples)
    start_idx = progress["next_index"]
    results = progress["results"]
    print(f"Resume from sample index: {start_idx}")
    skipped_count = 0
    current_shard_idx = None
    current_cache = None

    for sample_idx in base.tqdm(range(start_idx, total_samples), total=total_samples, initial=start_idx, desc="VGG clean ablation"):
        sample = samples[sample_idx]
        shard_idx = sample["shard_idx"]
        row_idx = sample["row_idx"]
        true_indices = sample["true_indices"]

        if current_shard_idx != shard_idx:
            current_cache = base.load_shard_cache(shard_paths[shard_idx])
            current_shard_idx = shard_idx

        audio_item = current_cache["audio"][row_idx]
        try:
            audio_path = base.write_audio_bytes(audio_item, sample_idx)
        except Exception:
            skipped_count += 1
            progress["next_index"] = sample_idx + 1
            base.save_progress(progress)
            continue

        try:
            t_audio_start = time.time()
            audio_emb_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = base.F.normalize(base.to_tensor(audio_emb_raw).to(base.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio_start) * 1000.0
        except Exception:
            skipped_count += 1
            progress["next_index"] = sample_idx + 1
            base.save_progress(progress)
            try:
                os.remove(audio_path)
            except OSError:
                pass
            continue

        cos_sim_orig = base.torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = base.torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:base.TOP_K]
        alpha_dynamic = base.instance_alpha(base.torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count = base.run_baseline(cos_sim_orig)
        results["Baseline"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Baseline"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Baseline"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = base.run_iknow(clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, get_tails)
        results["iKnow"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["iKnow"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["iKnow"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count = base.run_full2hop_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Full2Hop"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Full2Hop"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Full2Hop"]["prompts"].append(prompt_count)

        t = time.time()
        score, prompt_count, extras = base.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "dynamic"
        )
        results["Ours"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Ours"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Ours"]["prompts"].append(prompt_count)
        results["Ours"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Ours"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Ours"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = base.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, True, "original"
        )
        results["Selective2Hop_OriginalAgg"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Selective2Hop_OriginalAgg"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_OriginalAgg"]["prompts"].append(prompt_count)
        results["Selective2Hop_OriginalAgg"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_OriginalAgg"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_OriginalAgg"]["alphas"].append(alpha_dynamic)

        t = time.time()
        score, prompt_count, extras = base.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set,
            hop1_relations, hop2_relations, get_tails, prompt_map, alpha_dynamic, False, "dynamic"
        )
        results["Selective2Hop_NoLLM"]["ranks"].append(base.rank_of_true(score, true_indices))
        results["Selective2Hop_NoLLM"]["times"].append(audio_ms + (time.time() - t) * 1000.0)
        results["Selective2Hop_NoLLM"]["prompts"].append(prompt_count)
        results["Selective2Hop_NoLLM"]["hop2_activation_sample"].append(bool(extras["hop2_activated"]))
        results["Selective2Hop_NoLLM"]["hop2_activation_candidate"].append(float(extras["candidate_level_activation_rate"]))
        results["Selective2Hop_NoLLM"]["alphas"].append(alpha_dynamic)

        try:
            os.remove(audio_path)
        except OSError:
            pass

        progress["next_index"] = sample_idx + 1
        progress["results"] = results
        base.save_progress(progress)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid samples.")
    base.save_results(results, total_samples, completed=True)
    base.print_main_tables(results)


if __name__ == "__main__":
    base.OUTPUT_JSON = OUTPUT_JSON
    base.PROGRESS_JSON = PROGRESS_JSON
    base.TMP_AUDIO_DIR = TMP_AUDIO_DIR
    main()
