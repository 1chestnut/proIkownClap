# -*- coding: utf-8 -*-
import os

import torch

from tut28_common import run_tut_experiment

OUTPUT_DIR = "/data/zkx/zkx/iknow-audio/28.next/tut_prefilter"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results_prefilter.json")
PREFILTER_MARGIN = 0.03
MIN_KEEP = 2


def experiment_runner(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, context):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    for ci in top_indices:
        class_name = label_classes[ci]
        kg_ent = base.get_kg_entity(kg_classes[ci])
        tau = cos_sim_orig[ci].item() + base.RELATIVE_MARGIN
        h1_map = base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
        s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
        max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
        if s1.numel() > 0 and max_h1 >= tau:
            hop2_flags.append(False)
            all_scores = s1
            prompt_count += len(h1_map)
        else:
            hop2_flags.append(True)
            h2_prompts = base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
            s2 = base.score_prompt_list(clap_model, audio_embed, h2_prompts)
            all_scores = torch.cat([s1, s2 * base.DECAY_GAMMA]) if s2.numel() > 0 else s1
            prompt_count += len(h1_map) + len(h2_prompts)

        if all_scores.numel() == 0:
            continue
        threshold = float(cos_sim_orig[ci].item()) - PREFILTER_MARGIN
        kept = all_scores[all_scores >= threshold]
        if kept.numel() == 0:
            kept, _ = torch.topk(all_scores, min(MIN_KEEP, all_scores.numel()))
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, kept * base.LOGIT_SCALE])
        score[ci] = (torch.logsumexp(logits, dim=0) - base.np.log(logits.numel())) / base.LOGIT_SCALE

    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(base.np.mean(hop2_flags)) if hop2_flags else 0.0,
        "alpha_mean": 0.0,
    }
    return score, prompt_count, extras


if __name__ == "__main__":
    run_tut_experiment(
        exp_title="Starting TUT acoustic prefilter experiment",
        exp_label="KnowledgePrefilter",
        output_json=OUTPUT_JSON,
        experiment_runner=experiment_runner,
        extra_meta={"prefilter_margin": PREFILTER_MARGIN, "min_keep": MIN_KEEP},
    )
