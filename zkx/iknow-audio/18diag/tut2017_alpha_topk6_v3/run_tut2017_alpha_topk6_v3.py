# -*- coding: utf-8 -*-
import importlib.util, json, os, time
from pathlib import Path
ROOT = Path('/data/zkx/zkx/iknow-audio')
MODULE_PATH = str(ROOT / '17diag' / 'tut2017' / 'test_ablation.py')
OUTPUT_DIR = str(ROOT / '18diag' / 'tut2017_alpha_topk6_v3')
PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'progress.json')
RESULT_JSON = os.path.join(OUTPUT_DIR, 'results_adjustment.json')
REF_JSON = str(ROOT / '17diag' / 'tut2017' / 'results_ablation.json')
NEW_ALPHA_MIN = 0.2
NEW_ALPHA_MAX = 0.6
NEW_TOP_K = 6
RESULT_KEY = 'Ours_AlphaTopK6V3'

def load_module(path):
    spec = importlib.util.spec_from_file_location('diag_tut2017_alpha_topk6_v3', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
ab = load_module(MODULE_PATH)
ab.ALPHA_MIN = NEW_ALPHA_MIN
ab.ALPHA_MAX = NEW_ALPHA_MAX
ab.TOP_K = NEW_TOP_K
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def init_progress(total):
    return {'next_index': 0, 'total_samples': total, 'results': {RESULT_KEY: {'ranks': [], 'times': [], 'prompts': [], 'hop2_activation_sample': [], 'hop2_activation_candidate': []}}}

def load_progress(total):
    if not os.path.exists(PROGRESS_JSON):
        return init_progress(total)
    try:
        payload = json.loads(Path(PROGRESS_JSON).read_text(encoding='utf-8'))
    except Exception:
        return init_progress(total)
    fresh = init_progress(total)
    fresh['next_index'] = int(payload.get('next_index', 0))
    old = payload.get('results', {}).get(RESULT_KEY, {})
    for k in fresh['results'][RESULT_KEY]:
        fresh['results'][RESULT_KEY][k] = old.get(k, [])
    return fresh

def save_progress(progress):
    tmp = PROGRESS_JSON + '.tmp'
    Path(tmp).write_text(json.dumps(progress, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, PROGRESS_JSON)

def compute_stats(values):
    return {
        'avg_prompts': float(ab.np.mean(values['prompts'])) if values['prompts'] else 0.0,
        'avg_time_ms': float(ab.np.mean(values['times'])) if values['times'] else 0.0,
        'hop2_activation_sample': float(ab.np.mean(values['hop2_activation_sample'])) if values['hop2_activation_sample'] else 0.0,
        'hop2_activation_candidate': float(ab.np.mean(values['hop2_activation_candidate'])) if values['hop2_activation_candidate'] else 0.0,
    }

def save_results(results):
    ref = json.loads(Path(REF_JSON).read_text(encoding='utf-8'))
    vals = results[RESULT_KEY]
    metrics = ab.compute_metrics(vals['ranks'])
    payload = {
        'reference': ref,
        'adjusted': {
            'metrics': {'Hit@1': metrics[0], 'Hit@3': metrics[1], 'Hit@5': metrics[2], 'MRR': metrics[3]},
            'stats': compute_stats(vals),
            'alpha_min': NEW_ALPHA_MIN,
            'alpha_max': NEW_ALPHA_MAX,
            'top_k': NEW_TOP_K,
        },
    }
    Path(RESULT_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

@ab.torch.no_grad()
def main():
    print(f'Starting TUT2017 diagnostic: alpha=[{NEW_ALPHA_MIN}, {NEW_ALPHA_MAX}] + top_k={NEW_TOP_K}')
    dataset = ab.load_dataset()
    progress = load_progress(dataset['total'])
    start_index = progress['next_index']
    results = progress['results']
    print(f'Resume from sample index: {start_index}')
    prompt_map = ab.load_llm_prompts(ab.LLM_PROMPTS_PATH)
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
    for idx in ab.tqdm(range(start_index, len(samples)), desc=f'TUT2017 alpha+topk{NEW_TOP_K}'):
        sample = samples[idx]
        audio_path = sample['audio_path']
        true_indices = sample['true_indices']
        if not os.path.exists(audio_path) or not true_indices:
            progress['next_index'] = idx + 1
            if (idx + 1) % 50 == 0:
                save_progress(progress)
            continue
        try:
            t_audio = time.time()
            audio_embed_raw = clap_model.get_audio_embeddings([audio_path])
            audio_embed = ab.F.normalize(ab.to_tensor(audio_embed_raw).to(ab.DEVICE).float(), dim=-1)
            audio_ms = (time.time() - t_audio) * 1000.0
        except Exception:
            progress['next_index'] = idx + 1
            if (idx + 1) % 50 == 0:
                save_progress(progress)
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
        vals = results[RESULT_KEY]
        vals['ranks'].append(ab.rank_of_true(score, true_indices))
        vals['times'].append(audio_ms + (time.time() - t) * 1000.0)
        vals['prompts'].append(prompt_count)
        vals['hop2_activation_sample'].append(bool(extras['hop2_activated']))
        vals['hop2_activation_candidate'].append(float(extras['candidate_level_activation_rate']))
        progress['next_index'] = idx + 1
        if (idx + 1) % 50 == 0:
            save_progress(progress)
            save_results(results)
    save_progress(progress)
    save_results(results)

if __name__ == '__main__':
    main()
