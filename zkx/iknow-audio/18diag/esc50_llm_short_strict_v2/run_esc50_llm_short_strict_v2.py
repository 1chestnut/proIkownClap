# -*- coding: utf-8 -*-
import importlib.util, json, os, re, time
from pathlib import Path

ROOT = Path('/data/zkx/zkx/iknow-audio')
MODULE_PATH = str(ROOT / '17diag' / 'esc50' / 'test_ablation.py')
OUTPUT_DIR = str(ROOT / '18diag' / 'esc50_llm_short_strict_v2')
PROMPT_JSON = os.path.join(OUTPUT_DIR, 'llm_prompts_short_qwen_strict_v2.json')
PROMPT_PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'prompt_gen_progress.json')
EVAL_PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'eval_progress.json')
RESULT_JSON = os.path.join(OUTPUT_DIR, 'results_adjustment.json')
REF_JSON = str(ROOT / '17diag' / 'esc50' / 'results_ablation.json')
FILTERED_REF_JSON = str(ROOT / '18.??' / 'esc50_llm_filter' / 'results_adjustment.json')
QWEN_CANDIDATES = []
SAVE_EVERY_PROMPTS = 10
SAVE_EVERY_SAMPLES = 50
MAX_WORDS = 12
MIN_WORDS = 4
BANNED = ['however', 'therefore', 'graph', 'triple', 'relation', 'knowledge', 'model answer', 'validation', 'based solely', 'correct format', 'described as', 'typically', 'often']
SYSTEM_PROMPT = """You rewrite an audio knowledge graph triple into one short English sentence for audio-text matching.
Rules:
- Write exactly one short sentence.
- Use 6 to 12 words.
- Must include the original head phrase and the original tail phrase exactly.
- No explanation, no acoustic elaboration, no extra clauses.
- Do not mention graph, triple, relation, or knowledge.
- Output only the sentence."""
EXAMPLES = """Examples:
[airplane, belongs to class, vehicle] -> airplane sound belongs to the vehicle class.
[rain, perceived as, continuous] -> rain sound is perceived as continuous.
[dog bark, has parent, animal sound] -> dog bark is a kind of animal sound.
[glass breaking, event composed of, shattering] -> glass breaking sound includes shattering."""

def load_module(path):
    spec = importlib.util.spec_from_file_location('esc_ablation_module_strict_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ab = load_module(MODULE_PATH)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def choose_qwen_path():
    model_root = ROOT / 'data' / 'model'
    direct = model_root / '??'
    if direct.is_dir():
        return str(direct)
    candidates = []
    if model_root.exists():
        for pp in model_root.iterdir():
            if pp.is_dir() and pp.name != 'gpt2':
                candidates.append(str(pp))
    candidates.extend([path for path in QWEN_CANDIDATES if os.path.isdir(path)])
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'No local Qwen model found under {model_root}; scanned={candidates}')

def load_source_keys():
    prompt_root = next((pp for pp in ROOT.iterdir() if pp.is_dir() and pp.name.startswith('1.')), None)
    if prompt_root is None:
        raise FileNotFoundError('No prompt source root found')
    esc_dir = prompt_root / 'esc50'
    candidate = None
    for pp in esc_dir.iterdir():
        if pp.name.startswith('llm_prompts_acoustic') and pp.suffix == '.json':
            candidate = pp
            break
    if candidate is None:
        candidate = esc_dir / 'llm_prompts.json'
    data = json.loads(candidate.read_text(encoding='utf-8'))
    return list(data.keys())

def normalize_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"\' ')
    if 'Model answer:' in text:
        text = text.split('Model answer:')[-1].strip()
    text = text.split('\n')[0].strip()
    pieces = re.split(r'(?<=[.!?])\s+', text)
    if pieces:
        text = pieces[0].strip()
    return text

def fallback_prompt(head, rel, tail):
    h = head.strip(); t = tail.strip()
    if rel == 'belongs to class': return f'{h} sound belongs to the {t} class.'
    if rel == 'has parent': return f'{h} sound is a kind of {t} sound.'
    if rel == 'perceived as': return f'{h} sound is perceived as {t}.'
    if rel == 'event composed of': return f'{h} sound includes {t}.'
    if rel == 'has children': return f'{h} sound includes {t}.'
    return f'{h}, {t}'

def prompt_is_valid(text, head=None, tail=None):
    if not text:
        return False
    lower = text.lower()
    if any(token in lower for token in BANNED):
        return False
    if any(ch in text for ch in ['[', ']', '{', '}', ':', ';']):
        return False
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        return False
    wc = len(text.split())
    if wc < MIN_WORDS or wc > MAX_WORDS:
        return False
    compact = text.replace(' ', '')
    alpha_chars = sum(ch.isalpha() for ch in compact)
    if alpha_chars / max(1, len(compact)) < 0.78:
        return False
    tokens = text.split()
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'/-]*[.,!?]?", token) for token in tokens):
        return False
    if re.search(r'(.)\1{3,}', lower):
        return False
    if text.count(',') > 1:
        return False
    if head and head.lower() not in lower:
        return False
    if tail and tail.lower() not in lower:
        return False
    return True

def load_tokenizer_and_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qwen_path = choose_qwen_path()
    print(f'Loading local Qwen from: {qwen_path}')
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_path, dtype=ab.torch.float16, local_files_only=True).cuda()
    return tokenizer, model

def generate_short_prompt(tokenizer, model, head, rel, tail):
    prompt = SYSTEM_PROMPT.strip() + '\n\n' + EXAMPLES.strip() + f'\nNow rewrite: [{head}, {rel}, {tail}]\nAnswer:'
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    response = normalize_text(response)
    if not prompt_is_valid(response, head=head, tail=tail):
        response = fallback_prompt(head, rel, tail)
    return response

def load_existing_prompt_bank():
    if not os.path.exists(PROMPT_JSON):
        return {}
    return json.loads(Path(PROMPT_JSON).read_text(encoding='utf-8'))

def save_prompt_bank(prompt_bank, progress):
    Path(PROMPT_JSON + '.tmp').write_text(json.dumps(prompt_bank, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(PROMPT_JSON + '.tmp', PROMPT_JSON)
    Path(PROMPT_PROGRESS_JSON + '.tmp').write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(PROMPT_PROGRESS_JSON + '.tmp', PROMPT_PROGRESS_JSON)

def build_short_prompt_bank():
    keys = load_source_keys()
    prompt_bank = load_existing_prompt_bank()
    tokenizer, model = load_tokenizer_and_model()
    fallback_count = 0
    kept_valid_count = 0
    repaired_invalid_count = 0
    print(f'Building strict short prompt bank across {len(keys)} entries')
    for idx in ab.tqdm(range(len(keys)), desc='ESC50 strict-short generation'):
        key = keys[idx]
        try:
            head, rel, tail = [x.strip() for x in key.split('||')]
        except ValueError:
            continue
        generated = generate_short_prompt(tokenizer, model, head, rel, tail)
        if generated == fallback_prompt(head, rel, tail):
            fallback_count += 1
        else:
            kept_valid_count += 1
        prompt_bank[key] = generated
        if (idx + 1) % SAVE_EVERY_PROMPTS == 0:
            save_prompt_bank(prompt_bank, {'total_keys': len(keys), 'completed_keys': idx + 1, 'fallback_count': fallback_count, 'kept_valid_count': kept_valid_count, 'repaired_invalid_count': repaired_invalid_count})
    progress = {'total_keys': len(keys), 'completed_keys': len(keys), 'fallback_count': fallback_count, 'kept_valid_count': kept_valid_count, 'repaired_invalid_count': repaired_invalid_count}
    save_prompt_bank(prompt_bank, progress)
    return prompt_bank, progress

def init_eval_progress(total):
    return {'next_index': 0, 'total_samples': total, 'results': {'Adjusted_ShortLLM_StrictV2': {'ranks': [], 'times': [], 'prompts': [], 'hop2_activation_sample': [], 'hop2_activation_candidate': []}}}

def load_eval_progress(total):
    if not os.path.exists(EVAL_PROGRESS_JSON):
        return init_eval_progress(total)
    try:
        payload = json.loads(Path(EVAL_PROGRESS_JSON).read_text(encoding='utf-8'))
    except Exception:
        return init_eval_progress(total)
    fresh = init_eval_progress(total)
    fresh['next_index'] = int(payload.get('next_index', 0))
    old = payload.get('results', {}).get('Adjusted_ShortLLM_StrictV2', {})
    for k in fresh['results']['Adjusted_ShortLLM_StrictV2']:
        fresh['results']['Adjusted_ShortLLM_StrictV2'][k] = old.get(k, [])
    return fresh

def save_eval_progress(progress):
    Path(EVAL_PROGRESS_JSON + '.tmp').write_text(json.dumps(progress, ensure_ascii=False), encoding='utf-8')
    os.replace(EVAL_PROGRESS_JSON + '.tmp', EVAL_PROGRESS_JSON)

def compute_eval_stats(values):
    return {
        'avg_prompts': float(ab.np.mean(values['prompts'])) if values['prompts'] else 0.0,
        'avg_time_ms': float(ab.np.mean(values['times'])) if values['times'] else 0.0,
        'hop2_activation_sample': float(ab.np.mean(values['hop2_activation_sample'])) if values['hop2_activation_sample'] else 0.0,
        'hop2_activation_candidate': float(ab.np.mean(values['hop2_activation_candidate'])) if values['hop2_activation_candidate'] else 0.0,
    }

def save_results(results, prompt_progress):
    ref = json.loads(Path(REF_JSON).read_text(encoding='utf-8'))
    filtered_ref = json.loads(Path(FILTERED_REF_JSON).read_text(encoding='utf-8')) if os.path.exists(FILTERED_REF_JSON) else None
    vals = results['Adjusted_ShortLLM_StrictV2']
    metrics = ab.compute_metrics(vals['ranks'])
    payload = {
        'reference': ref,
        'filtered_reference': filtered_ref,
        'adjusted': {
            'metrics': {'Hit@1': metrics[0], 'Hit@3': metrics[1], 'Hit@5': metrics[2], 'MRR': metrics[3]},
            'stats': compute_eval_stats(vals),
            'prompt_file': PROMPT_JSON,
            'prompt_generation': prompt_progress,
            'template': 'short_low_freedom_non_expansive_qwen_strict_v2',
            'max_words': MAX_WORDS,
        },
    }
    Path(RESULT_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

@ab.torch.no_grad()
def main():
    print('Starting ESC50 diagnostic: strict short-template LLM prompt bank v2')
    prompt_map, prompt_progress = build_short_prompt_bank()
    dataset = ab.load_dataset()
    progress = load_eval_progress(dataset['total'])
    print(f"Resume ESC50 eval from sample index: {progress['next_index']}")
    clap_model = ab.CLAP(version='2023', use_cuda=ab.torch.cuda.is_available())
    with ab.warnings.catch_warnings():
        ab.warnings.simplefilter('ignore')
        kge_model = ab.torch.load(os.path.join(ab.KGE_MODEL_DIR, 'trained_model.pkl'), map_location=ab.DEVICE)
    kge_model.eval()
    training_factory = ab.TriplesFactory.from_path(ab.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in ab.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in ab.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = ab.build_tail_predictor(kge_model, training_factory)
    text_embeds = ab.F.normalize(ab.to_tensor(clap_model.get_text_embeddings(dataset['label_classes'])).to(ab.DEVICE).float(), dim=-1)
    samples = list(ab.iter_samples(dataset))
    results = progress['results']
    for idx in ab.tqdm(range(progress['next_index'], len(samples)), desc='ESC50 strict-short-LLM-v2'):
        sample = samples[idx]
        audio_path = sample['audio_path']
        true_indices = sample['true_indices']
        if not os.path.exists(audio_path) or not true_indices:
            progress['next_index'] = idx + 1
            if (idx + 1) % SAVE_EVERY_SAMPLES == 0:
                save_eval_progress(progress)
            continue
        try:
            t_audio = time.time()
            audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = ab.F.normalize(ab.to_tensor(audio_embed_raw).to(ab.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio) * 1000.0
        except Exception:
            progress['next_index'] = idx + 1
            if (idx + 1) % SAVE_EVERY_SAMPLES == 0:
                save_eval_progress(progress)
            continue
        cos_sim_orig = ab.torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = ab.torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:ab.TOP_K]
        alpha_dynamic = ab.instance_alpha(ab.torch.max(cos_sim_orig).item())
        t = time.time()
        score, prompt_count, extras = ab.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices,
            dataset['label_classes'], dataset['kg_classes'], dataset['class_labels_set'],
            hop1_relations, hop2_relations, get_tails, prompt_map,
            alpha_dynamic, True, 'dynamic'
        )
        vals = results['Adjusted_ShortLLM_StrictV2']
        vals['ranks'].append(ab.rank_of_true(score, true_indices))
        vals['times'].append(audio_ms + (time.time() - t) * 1000.0)
        vals['prompts'].append(prompt_count)
        vals['hop2_activation_sample'].append(bool(extras['hop2_activated']))
        vals['hop2_activation_candidate'].append(float(extras['candidate_level_activation_rate']))
        progress['next_index'] = idx + 1
        if (idx + 1) % SAVE_EVERY_SAMPLES == 0:
            save_eval_progress(progress)
            save_results(results, prompt_progress)
    save_eval_progress(progress)
    save_results(results, prompt_progress)

if __name__ == '__main__':
    main()
