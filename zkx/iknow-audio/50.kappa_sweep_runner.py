import argparse
import json
import os
from pathlib import Path
from typing import Any
import types
import re

import numpy as np


def _replace_assignment(source: str, name: str, value: str) -> str:
    pattern = rf'^{name}\s*=\s*.*$'
    replacement = f'{name} = "{value}"'
    return re.sub(pattern, replacement, source, flags=re.MULTILINE)


def load_module(module_path: str, gpu_id: int, kappa: float, output_json: str):
    # Force the imported ablation script to see only one physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    source = Path(module_path).read_text(encoding="utf-8")
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"', f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"')
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"')
    source = source.replace("LOGIT_SCALE = 100.0", f"LOGIT_SCALE = {float(kappa)}")
    progress_json = str(Path(output_json).with_name(f"{Path(output_json).stem}_progress.json"))
    source = _replace_assignment(source, "OUTPUT_JSON", output_json.replace("\\", "/"))
    source = _replace_assignment(source, "PROGRESS_JSON", progress_json.replace("\\", "/"))
    module = types.ModuleType("kappa50_base_module")
    module.__file__ = module_path
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


def build_payload(module: Any, results: dict, kappa: float, module_path: str) -> dict:
    metrics = {name: module.compute_metrics(results[name]["ranks"]) for name in results}
    summary = {}
    for name in results:
        entry = {
            "Hit@1": float(metrics[name][0]),
            "Hit@3": float(metrics[name][1]),
            "Hit@5": float(metrics[name][2]),
            "MRR": float(metrics[name][3]),
            "avg_prompts": float(np.mean(results[name]["prompts"])) if results[name].get("prompts") else 0.0,
            "avg_time_ms": float(np.mean(results[name]["times"])) if results[name].get("times") else 0.0,
        }
        if "triggers" in results[name]:
            entry["trigger_rate"] = float(np.mean(results[name]["triggers"]))
        if "alphas" in results[name]:
            entry["avg_alpha"] = float(np.mean(results[name]["alphas"]))
        summary[name] = entry
    return {
        "module_path": module_path,
        "kappa": float(kappa),
        "results": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    module = load_module(args.module_path, args.gpu_id, args.kappa, args.output_json)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_name = None
    if hasattr(module, "print_final_tables"):
        target_name = "print_final_tables"
    elif hasattr(module, "print_main_tables"):
        target_name = "print_main_tables"
    else:
        raise AttributeError("No printable result hook found (expected print_final_tables or print_main_tables)")

    original_print = getattr(module, target_name)

    def patched_print(results):
        payload = build_payload(module, results, args.kappa, args.module_path)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return original_print(results)

    setattr(module, target_name, patched_print)
    module.main()


if __name__ == "__main__":
    main()
