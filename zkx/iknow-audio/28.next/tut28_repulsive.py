# -*- coding: utf-8 -*-
import os

import torch

from tut28_common import run_tut_experiment

OUTPUT_DIR = "/data/zkx/zkx/iknow-audio/28.next/tut_repulsive"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results_repulsive.json")
REPULSIVE_BETA = 0.35
REPULSIVE_MARGIN = 0.01


def candidate_originalagg_score(base, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map):
    class_name = label_classes[ci]
    kg_ent = base.get_kg_entity(kg_classes[ci])
    tau = cos_sim_orig[ci].item() + base.RELATIVE_MARGIN
    h1_map = base.build_h1_map(kg_ent, class_name, class_labels_set, hop1_relations, get_tails, prompt_map)
    s1 = base.score_prompt_list(clap_model, audio_embed, list(h1_map.values()))
    max_h1 = torch.max(s1).item() if s1.numel() > 0 else -999.0
    if s1.numel() > 0 and max_h1 >= tau:
        logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, s1 * base.LOGIT_SCALE])
        score = (torch.logsumexp(logits, dim=0) - base.np.log(logits.numel())) / base.LOGIT_SCALE
        return score, len(h1_map), False
    h2_prompts = base.build_h2_prompts(h1_map, class_name, class_labels_set, hop2_relations, get_tails, prompt_map)
    s2 = base.score_prompt_list(clap_model, audio_embed, h2_prompts)
    all_scores = torch.cat([s1, s2 * base.DECAY_GAMMA]) if s2.numel() > 0 else s1
    if all_scores.numel() == 0:
        return cos_sim_orig[ci], len(h1_map) + len(h2_prompts), True
    logits = torch.cat([cos_sim_orig[ci].unsqueeze(0) * base.LOGIT_SCALE, all_scores * base.LOGIT_SCALE])
    score = (torch.logsumexp(logits, dim=0) - base.np.log(logits.numel())) / base.LOGIT_SCALE
    return score, len(h1_map) + len(h2_prompts), True


def experiment_runner(base, clap_model, audio_embed, cos_sim_orig, top_indices, label_classes, kg_classes, class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map, context):
    score = cos_sim_orig.clone()
    prompt_count = 0
    hop2_flags = []
    candidate_scores = {}
    for ci in top_indices:
        s_i, used_prompts, hop2_flag = candidate_originalagg_score(
            base, clap_model, audio_embed, cos_sim_orig, ci, label_classes, kg_classes,
            class_labels_set, hop1_relations, hop2_relations, get_tails, prompt_map
        )
        candidate_scores[ci] = s_i
        prompt_count += used_prompts
        hop2_flags.append(hop2_flag)

    for ci in top_indices:
        own = candidate_scores[ci]
        competitors = [float(candidate_scores[cj].item()) for cj in top_indices if cj != ci]
        comp = max(competitors) if competitors else float(own.item())
        penalty = max(0.0, comp - float(own.item()) + REPULSIVE_MARGIN)
        score[ci] = own - (REPULSIVE_BETA * penalty)

    extras = {
        "hop2_activated": any(hop2_flags),
        "candidate_level_activation_rate": float(base.np.mean(hop2_flags)) if hop2_flags else 0.0,
        "alpha_mean": 0.0,
    }
    return score, prompt_count, extras


if __name__ == "__main__":
    run_tut_experiment(
        exp_title="Starting TUT competitor-aware repulsive fusion experiment",
        exp_label="RepulsiveFusion",
        output_json=OUTPUT_JSON,
        experiment_runner=experiment_runner,
        extra_meta={"repulsive_beta": REPULSIVE_BETA, "repulsive_margin": REPULSIVE_MARGIN},
    )
