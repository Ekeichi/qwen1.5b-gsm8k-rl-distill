"""
Tableau recapitulatif : baseline vs checkpoints.
Lit results/eval/<nom>_greedy.jsonl et results/eval/<nom>_maj16.jsonl (sorties de src/eval_gsm8k.py).

Usage (depuis la racine du depot) :
    python src/compare.py | tee results/summary.txt
Les checkpoints absents sont ignores.
"""
import json
import os
import statistics as st

RESULTS = "results/eval"
NAMES = ["baseline", "ckpt100", "ckpt200", "ckpt300", "ckpt400", "final", "sft"]
NAMES += ["sft_grpo100", "sft_grpo200", "sft_grpo300", "sft_grpo400", "sft_grpo_final"]  # GRPO depuis le SFT


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def per_question(rs):
    return [sum(r["correct"]) / len(r["correct"]) for r in rs]


base_m = load(f"{RESULTS}/baseline_maj16.jsonl")
base_pq = per_question(base_m) if base_m else None

header = f"{'modele':<15} {'greedy':>8} {'pass@1':>8} {'maj@16':>8}   {'gain pass@1 vs baseline (IC95)':<32}"
print(header)
print("-" * len(header))

for name in NAMES:
    g = load(f"{RESULTS}/{name}_greedy.jsonl")
    m = load(f"{RESULTS}/{name}_maj16.jsonl")
    if g is None and m is None:
        continue

    greedy = f"{100 * st.mean(r['correct'][0] for r in g):.2f}" if g else "-"
    p1 = maj = gain = "-"
    if m:
        pq = per_question(m)
        p1 = f"{100 * st.mean(pq):.2f}"
        maj = f"{100 * st.mean(float(r['maj_correct']) for r in m):.2f}"
        if name != "baseline" and base_pq and len(base_pq) == len(pq):
            d = [y - x for x, y in zip(base_pq, pq)]
            se = st.stdev(d) / len(d) ** 0.5
            gain = f"{100 * st.mean(d):+.2f} pts (+-{196 * se:.2f})"
    print(f"{name:<15} {greedy:>8} {p1:>8} {maj:>8}   {gain:<32}")

print()
print("Lecture : un gain est reel si l'intervalle (+-) n'inclut pas 0.")
print("Le greedy vLLM varie d'environ +-0.2 pt d'un run a l'autre sur le meme modele.")
