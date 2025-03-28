import os
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
import timeout_decorator

import numpy as np
import torch

import ray

from src.eval_utils.data import load_datasets, load_jsonl, save_jsonl, save_json
from src.eval_utils.grader import math_equal
from src.eval_utils.parser import parse_ground_truth, extract_and_strip
from src.eval_utils.vote import AGG_FN_MAP
from src.eval_utils.gen_utils import Generator, PRMPredictor


@ray.remote
class RemoteGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote
class RemotePRMPredictor(PRMPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote
def remote_math_equal(pred, gt_ans):
    try:
        return math_equal(pred, gt_ans)
    except Exception:
        return False


@ray.remote
class RemoteMathEvaluator:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def evaluate(self, sample):
        sample["gt_ans"] = parse_ground_truth(sample, sample["dataset"])[-1]

        for item in sample["generation"]:
            item["pred"] = extract_and_strip(
                item["response"], data_name=sample["dataset"]
            )
            try:
                item["correct"] = ray.get(
                    remote_math_equal.remote(item["pred"], sample["gt_ans"]),
                    timeout=self.timeout
                )
            except:
                item["correct"] = False

        return sample


def compute_metrics_fn(eval_results, k, agg_method):
    final_results = []
    for sample in eval_results:
        eval_samples = [
            {
                "dataset": sample["dataset"],
                "ans": generation["pred"],
                "scores": generation.get("step_rewards", None),
                "correct": generation["correct"],
            }
            for generation in sample["generation"][:k]
        ]
        final_result = AGG_FN_MAP[agg_method](eval_samples)
        final_results.append(final_result)

    dataset_counts = defaultdict(int)
    dataset_correct = defaultdict(int)
    for result in final_results:
        dataset = result["dataset"]
        dataset_counts[dataset] += 1
        if result["correct"]:
            dataset_correct[dataset] += 1

    metrics = []
    for dataset, count in dataset_counts.items():
        correct = dataset_correct[dataset]
        metrics.append(
            {
                "dataset": dataset,
                "total": count,
                "corrcet": correct,
                "accuracy": correct / count,
            }
        )
    average_accuracy = np.mean([float(metric["accuracy"]) for metric in metrics])
    metrics.append({"Average": average_accuracy})

    return metrics


# Step 0: Parameters
ray.init()
num_instances = torch.cuda.device_count()
lm_args = {
    "model_path": "Qwen/Qwen2.5-Math-7B-Instruct",
    "prompt_template": "qwen25-math-cot",
    "num_sequence": 8,
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.8,
    "tensor_parallel_size": 1,
}
rm_args = {
    "model_path": "/path/to/process_reward_model",
}
output_dir = f"{rm_args['model_path']}/best-of-n"
datasets = [
    "gsm8k/test",
    "math/test",
    "college_math/test",
    "olympiadbench/test",
]
generate = True
evaluate = True
reward_score = True
compute_metrics = True

os.makedirs(output_dir, exist_ok=True)

# Step 1: load datasets
datasets = load_datasets(datasets)

# Step 2: LM Generation
lm_output_path = os.path.join(output_dir, "generations.jsonl")
if generate:
    batch_size = min(10000, len(datasets) // num_instances + 1)
    llm_actors = [
        RemoteGenerator.options(num_gpus=1, num_cpus=1).remote(**lm_args)
        for _ in range(num_instances)
    ]
    dataset_batches = [
        datasets[i : i + batch_size] for i in range(0, len(datasets), batch_size)
    ]
    futures = ray.util.ActorPool(llm_actors).map_unordered(
        lambda actor, batch_data: actor.generate_batch.remote(batch_data),
        dataset_batches,
    )
    completed_futures = []
    for batch_idx, batch_results in enumerate(futures):
        completed_futures.extend(batch_results)
    save_jsonl(completed_futures, lm_output_path)

# Step 3: Evaluating
eval_output_path = os.path.join(output_dir, "eval_results.jsonl")
if evaluate:
    datasets = load_jsonl(lm_output_path)
    num_cpus = 16
    evaluators = [
        RemoteMathEvaluator.options(num_cpus=1).remote() for _ in range(num_cpus)
    ]
    futures = ray.util.ActorPool(evaluators).map_unordered(
        lambda actor, sample: actor.evaluate.remote(sample), datasets
    )
    results = []
    for result in tqdm(futures, total=len(datasets), desc="Evaluating samples"):
        results.append(result)
    save_jsonl(results, eval_output_path)

# Step 4: Reward Filtering
rm_output_path = os.path.join(output_dir, "prm_results.jsonl")
if reward_score:
    datasets = load_jsonl(eval_output_path)
    llm_actors = [
        RemotePRMPredictor.options(num_gpus=1, num_cpus=1).remote(**rm_args)
        for _ in range(num_instances)
    ]
    futures = ray.util.ActorPool(llm_actors).map_unordered(
        lambda actor, batch_data: actor.score.remote(batch_data),
        datasets,
    )
    results = []
    for result in tqdm(futures, total=len(datasets), desc="Reward Judging"):
        results.append(result)
    save_jsonl(results, rm_output_path)

# Step 5: Compute Metric
metrics_output_path = os.path.join(output_dir, "metrics.txt")
if compute_metrics:
    if os.path.exists(rm_output_path):
        datasets = load_jsonl(rm_output_path)
        agg_fn_list = list(AGG_FN_MAP.keys())
    else:
        datasets = load_jsonl(eval_output_path)
        agg_fn_list = ["pass", "majority_vote"]

    all_results = {}
    max_k = len(datasets[0]["generation"])
    for agg_method in agg_fn_list:
        results = []
        for k in [1, 2, 4, 8, 16, 32, 64]:
            if k > max_k:
                break
            metrics = compute_metrics_fn(datasets, k=k, agg_method=agg_method)
            result = {"k": k}
            result.update({metric["dataset"]: metric["accuracy"] for metric in metrics[:-1]})
            result.update(metrics[-1])
            results.append(result)

        all_results[agg_method] = results

    with open(metrics_output_path, "w") as f:
        for agg_method, result in all_results.items():
            f.write(f"{agg_method}:" + "\n\n")
            f.write(tabulate(result, headers="keys", tablefmt="grid", floatfmt=".4f") + "\n\n\n")

ray.shutdown()
