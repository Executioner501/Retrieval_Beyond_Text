"""
Deterministic stratified 50/50 split of evaluation_set_v2.json by (question_type, chat).
Generates data/benchmarks/eval_dev.json and data/benchmarks/eval_test.json.
"""

from collections import Counter, defaultdict
import json
import os
import random


def split_eval(
    input_path: str = "data/benchmarks/evaluation_set_v2.json",
    dev_path: str = "data/benchmarks/eval_dev.json",
    test_path: str = "data/benchmarks/eval_test.json",
    seed: int = 42,
):
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Group by stratum: (question_type, chat)
    groups = defaultdict(list)
    for q in items:
        key = (q.get("question_type", "unknown"), q.get("chat", "unknown"))
        groups[key].append(q)

    rng = random.Random(seed)
    dev_items = []
    test_items = []

    odd_toggle = True
    for key in sorted(groups.keys()):
        group = sorted(groups[key], key=lambda x: str(x["id"]))
        rng.shuffle(group)
        n = len(group)
        if n % 2 == 1:
            if odd_toggle:
                n_dev = n // 2 + 1
                odd_toggle = False
            else:
                n_dev = n // 2
                odd_toggle = True
        else:
            n_dev = n // 2

        dev_items.extend(group[:n_dev])
        test_items.extend(group[n_dev:])

    # Sort by ID deterministically
    dev_items.sort(key=lambda x: str(x["id"]))
    test_items.sort(key=lambda x: str(x["id"]))

    os.makedirs(os.path.dirname(dev_path), exist_ok=True)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_items, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2, ensure_ascii=False)

    print(f"Total items: {len(items)} -> Dev: {len(dev_items)}, Test: {len(test_items)}")

    dev_counts = Counter(q.get("question_type") for q in dev_items)
    test_counts = Counter(q.get("question_type") for q in test_items)

    all_types = sorted(set(dev_counts.keys()) | set(test_counts.keys()))
    print("\n=== Split Counts by Question Type ===")
    print(f"{'Question Type':<25} {'Dev':<8} {'Test':<8} {'Total':<8}")
    print("-" * 50)
    for qt in all_types:
        d = dev_counts.get(qt, 0)
        t = test_counts.get(qt, 0)
        print(f"{qt:<25} {d:<8} {t:<8} {d + t:<8}")

    return dev_items, test_items


if __name__ == "__main__":
    split_eval()
