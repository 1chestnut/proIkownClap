# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import re
import time
from pathlib import Path

MODULE_PATH = '/data/zkx/zkx/iknow-audio/17.消融/esc50/test_ablation.py'
OUTPUT_DIR = '/data/zkx/zkx/iknow-audio/18.调整/esc50_llm_short'
PROMPT_JSON = os.path.join(OUTPUT_DIR, 'llm_prompts_short_qwen.json')
PROMPT_PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'prompt_gen_progress.json')
EVAL_PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'eval_progress.json')
RESULT_JSON = os.path.join(OUTPUT_DIR, 'results_adjustment.json')
REF_JSON = '/data/zkx/zkx/iknow-audio/17.消融/esc50/results_ablation.json'
FILTERED_REF_JSON = '/data/zkx/zkx/iknow-audio/18.调整/esc50_llm_filter/results_adjustment.json'
SOURCE_KEYS_JSON = '/data/zkx/zkx/iknow-audio/1.单跳+提示词扩展+0.85权重/esc50/llm_prompts_acoustic.json（2）'
QWEN_CANDIDATES = [
    '/data/zkx/zkx/iknow-audio/data/model/千问',
    '/data/zkx/zkx/iknow-audio/data/model/千问模型',
]
SAVE_EVERY_PROMPTS = 10
SAVE_EVERY_SAMPLES = 50
MAX_WORDS = 16
MIN_WORDS = 4
BANNED = [
    'however', 'therefore', 'graph', 'triple', 'relation', 'knowledge',
    'model answer', 'validation', 'based solely', 'correct format'
]
SYSTEM_PROMPT = '''You rewrite an audio knowledge graph triple into one short English prompt for audio-text matching.
Rules:
- Write exactly one short sentence.
- Keep it concise, about 8 to 14 words when possible.
- Mention both the sound class and the relation target.
- No explanation, no acoustic elaboration, no extra clauses.
- Do not mention graph, triple, relation, or knowledge.
- Use plain present tense.
- Output only the sentence.'''
EXAMPLES = '''Examples:
[airplane, belongs to class, vehicle] -> airplane sound belongs to the vehicle class.
[rain, perceived as, continuous] -> rain sound is perceived as continuous.
[dog bark, has parent, animal sound] -> dog bark is a kind of animal sound.
[glass breaking, event composed of, shattering] -> glass breaking sound includes shattering.
'''


def load_module(path):
    spec = importlib.util.spec_from_file_location('esc_ablation_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = load_module(MODULE_PATH)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def choose_qwen_path():
    for path in QWEN_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'No local Qwen model found in {QWEN_CANDIDATES}')


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
    h = head.strip()
    t = tail.strip()
    if rel == 'belongs to class':
        return f'{h} sound belongs to the {t} class.'
    if rel == 'has parent':
        return f'{h} sound is a kind of {t} sound.'
    if rel == 'perceived as':
        return f'{h} sound is perceived as {t}.'
    if rel == 'event composed of':
        return f'{h} sound includes {t}.'
    if rel == 'has children':
        return f'{h} sound includes {t}.'
    return f'{h}, {t}'


def prompt_is_valid(text):
    if not text:
        return False
    lower = text.lower()
    if any(token in lower for token in BANNED):
        return False
    if any(ch in text for ch in ['[', ']', '{', '}', ':']):
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
    if alpha_chars / max(1, len(compact)) < 0.72:
        return False
    tokens = text.split()
    weird_tokens = 0
    for token in tokens:
        if not re.fullmatch(r"[A-Za-z][A-Za-z'/-]*[.,!?]?", token):
            weird_tokens += 1
    if weird_tokens > 0:
        return False
    if re.search(r'(.)\1{4,}', lower):
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
    prompt = (
        SYSTEM_PROMPT.strip()
        + '\n\n'
        + EXAMPLES.strip()
        + '\nNow rewrite: [' + head + ', ' + rel + ', ' + tail + ']\nAnswer:'
    )
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    response = normalize_text(response)
    if not prompt_is_valid(response):
        response = fallback_prompt(head, rel, tail)
    return response


def load_source_keys():
    with open(SOURCE_KEYS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return list(data.keys())


def load_existing_prompt_bank():
    if not os.path.exists(PROMPT_JSON):
        return {}
    with open(PROMPT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_prompt_bank(prompt_bank, progress):
    tmp_prompt = PROMPT_JSON + '.tmp'
    with open(tmp_prompt, 'w', encoding='utf-8') as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)
    os.replace(tmp_prompt, PROMPT_JSON)
    tmp_prog = PROMPT_PROGRESS_JSON + '.tmp'
    with open(tmp_prog, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp_prog, PROMPT_PROGRESS_JSON)


def build_short_prompt_bank():
    keys = load_source_keys()
    prompt_bank = load_existing_prompt_bank()
    progress = {
        'total_keys': len(keys),
        'completed_keys': 0,
        'fallback_count': 0,
        'kept_valid_count': 0,
        'repaired_invalid_count': 0,
    }

    need_generation = any(key not in prompt_bank for key in keys)
    tokenizer = None
    model = None
    if need_generation:
        tokenizer, model = load_tokenizer_and_model()
    fallback_count = 0
    kept_valid_count = 0
    repaired_invalid_count = 0
    print(f'Validating short prompt bank across {len(keys)} entries')
    for idx in ab.tqdm(range(len(keys)), desc='ESC50 short-prompt generation'):
        key = keys[idx]
        try:
            head, rel, tail = [x.strip() for x in key.split('||')]
        except ValueError:
            continue
        existing = prompt_bank.get(key)
        if isinstance(existing, str):
            normalized = normalize_text(existing)
            if prompt_is_valid(normalized):
                prompt_bank[key] = normalized
                kept_valid_count += 1
            else:
                prompt_bank[key] = fallback_prompt(head, rel, tail)
                fallback_count += 1
                repaired_invalid_count += 1
        else:
            generated = generate_short_prompt(tokenizer, model, head, rel, tail)
            if generated == fallback_prompt(head, rel, tail):
                fallback_count += 1
            else:
                kept_valid_count += 1
            prompt_bank[key] = generated
        if (idx + 1) % SAVE_EVERY_PROMPTS == 0:
            progress = {
                'total_keys': len(keys),
                'completed_keys': idx + 1,
                'fallback_count': fallback_count,
                'kept_valid_count': kept_valid_count,
                'repaired_invalid_count': repaired_invalid_count,
            }
            save_prompt_bank(prompt_bank, progress)
    progress = {
        'total_keys': len(keys),
        'completed_keys': len(keys),
        'fallback_count': fallback_count,
        'kept_valid_count': kept_valid_count,
        'repaired_invalid_count': repaired_invalid_count,
    }
    save_prompt_bank(prompt_bank, progress)
    return prompt_bank, progress


def init_eval_progress(total):
    return {
        'next_index': 0,
        'total_samples': total,
        'results': {
            'Adjusted_ShortLLM': {
                'ranks': [],
                'times': [],
                'prompts': [],
                'hop2_activation_sample': [],
                'hop2_activation_candidate': [],
            }
        },
    }


def load_eval_progress(total):
    if not os.path.exists(EVAL_PROGRESS_JSON):
        return init_eval_progress(total)
    try:
        with open(EVAL_PROGRESS_JSON, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return init_eval_progress(total)
    fresh = init_eval_progress(total)
    fresh['next_index'] = int(payload.get('next_index', 0))
    old = payload.get('results', {}).get('Adjusted_ShortLLM', {})
    for k in fresh['results']['Adjusted_ShortLLM']:
        fresh['results']['Adjusted_ShortLLM'][k] = old.get(k, [])
    return fresh


def save_eval_progress(progress):
    tmp = EVAL_PROGRESS_JSON + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False)
    os.replace(tmp, EVAL_PROGRESS_JSON)


def compute_stats(values):
    return {
        'avg_prompts': float(ab.np.mean(values['prompts'])) if values['prompts'] else 0.0,
        'avg_time_ms': float(ab.np.mean(values['times'])) if values['times'] else 0.0,
        'hop2_activation_sample': float(ab.np.mean(values['hop2_activation_sample'])) if values['hop2_activation_sample'] else 0.0,
        'hop2_activation_candidate': float(ab.np.mean(values['hop2_activation_candidate'])) if values['hop2_activation_candidate'] else 0.0,
    }


def save_results(results, prompt_progress):
    with open(REF_JSON, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    with open(FILTERED_REF_JSON, 'r', encoding='utf-8') as f:
        filtered_ref = json.load(f)
    vals = results['Adjusted_ShortLLM']
    metrics = ab.compute_metrics(vals['ranks'])
    stats = compute_stats(vals)
    payload = {
        'reference': ref,
        'filtered_reference': filtered_ref,
        'adjusted': {
            'metrics': {
                'Hit@1': metrics[0],
                'Hit@3': metrics[1],
                'Hit@5': metrics[2],
                'MRR': metrics[3],
            },
            'stats': stats,
            'prompt_file': PROMPT_JSON,
            'prompt_generation': prompt_progress,
            'template': 'short_low_freedom_non_expansive_qwen',
        },
    }
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_compare(results, prompt_progress):
    with open(REF_JSON, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    with open(FILTERED_REF_JSON, 'r', encoding='utf-8') as f:
        filtered_ref = json.load(f)
    ref_ours = ref['metrics']['Ours']
    ref_nollm = ref['metrics']['Selective2Hop_NoLLM']
    ref_filtered = filtered_ref['adjusted']['metrics']
    vals = results['Adjusted_ShortLLM']
    m = ab.compute_metrics(vals['ranks'])
    s = compute_stats(vals)
    print('\n' + '=' * 160)
    print(f"{'ESC50 LLM Diagnostic':<24} | {'NoLLM':<12} | {'Current_Ours':<14} | {'Filtered_AcousticLLM':<22} | {'Short_Template_LLM':<20}")
    print('-' * 160)
    for idx, name in enumerate(['Hit@1', 'Hit@3', 'Hit@5', 'MRR']):
        print(f"{name:<24} | {ref_nollm[name]:<12.2f} | {ref_ours[name]:<14.2f} | {ref_filtered[name]:<22.2f} | {m[idx]:<20.2f}")
    print('\n' + '=' * 160)
    print(
        f"Short_Template_LLM stats: prompts={s['avg_prompts']:.1f}, time={s['avg_time_ms']:.1f} ms, "
        f"activation(sample)={s['hop2_activation_sample']*100:.1f}%, activation(candidate)={s['hop2_activation_candidate']*100:.1f}%"
    )
    print(
        f"Prompt bank generation: completed={prompt_progress['completed_keys']}/{prompt_progress['total_keys']}, "
        f"fallback={prompt_progress['fallback_count']}"
    )


@ab.torch.no_grad()
def main():
    print('Starting ESC50 diagnostic: short-template LLM prompt bank')
    short_prompt_map, prompt_progress = build_short_prompt_bank()

    dataset = ab.load_dataset()
    progress = load_eval_progress(dataset['total'])
    start_index = progress['next_index']
    results = progress['results']
    print(f'Resume ESC50 eval from sample index: {start_index}')

    clap_model = ab.CLAP(version='2023', use_cuda=ab.torch.cuda.is_available())
    with ab.warnings.catch_warnings():
        ab.warnings.simplefilter('ignore')
        kge_model = ab.torch.load(os.path.join(ab.KGE_MODEL_DIR, 'trained_model.pkl'), map_location=ab.DEVICE)
    kge_model.eval()
    training_factory = ab.TriplesFactory.from_path(ab.TRAIN_TRIPLES_PATH)
    hop1_relations = [rel for rel in ab.HOP1_RELATIONS if rel in training_factory.relation_to_id]
    hop2_relations = [rel for rel in ab.HOP2_RELATIONS if rel in training_factory.relation_to_id]
    get_tails = ab.build_tail_predictor(kge_model, training_factory)

    text_embeds = ab.F.normalize(
        ab.to_tensor(clap_model.get_text_embeddings(dataset['label_classes'])).to(ab.DEVICE).float(), dim=-1
    )

    samples = list(ab.iter_samples(dataset))
    for idx in ab.tqdm(range(start_index, len(samples)), desc='ESC50 short-template-LLM'):
        sample = samples[idx]
        audio_path = sample['audio_path']
        true_indices = sample['true_indices']
        if not os.path.exists(audio_path) or not true_indices:
            progress['next_index'] = idx + 1
            if (idx + 1) % SAVE_EVERY_SAMPLES == 0:
                save_eval_progress(progress)
                save_results(results, prompt_progress)
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
                save_results(results, prompt_progress)
            continue

        cos_sim_orig = ab.torch.matmul(audio_embed, text_embeds.T).squeeze(0)
        sorted_indices = ab.torch.argsort(cos_sim_orig, descending=True).detach().cpu().numpy()
        top_indices = sorted_indices[:ab.TOP_K]
        alpha_dynamic = ab.instance_alpha(ab.torch.max(cos_sim_orig).item())

        t = time.time()
        score, prompt_count, extras = ab.run_selective_variant(
            clap_model, audio_embed, cos_sim_orig, top_indices,
            dataset['label_classes'], dataset['kg_classes'], dataset['class_labels_set'],
            hop1_relations, hop2_relations, get_tails, short_prompt_map,
            alpha_dynamic, True, 'dynamic'
        )
        vals = results['Adjusted_ShortLLM']
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
    print_compare(results, prompt_progress)


if __name__ == '__main__':
    main()

