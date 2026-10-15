"""
Taux de reussite par question du train GRPO, pour filtrer hors ligne les questions sans signal.

En GRPO, une question dont les num_generations reponses ont toutes la meme recompense (toutes justes
ou toutes fausses) a un avantage nul : elle coute generation + forward/backward pour un gradient nul,
et dilue en plus le gradient des autres (la loss DAPO de TRL normalise par TOUS les tokens du batch).

Ce script echantillonne les questions comme pendant l'entrainement (meme split, meme prompt,
meme temperature, meme budget de tokens, memes recompenses que train_grpo.py) et ecrit,
pour chaque question, le nombre de reponses correctes.

Usage :
    python src/filter_easy.py --model outputs/sft_distill --out data/pass_rates_sft.jsonl
    python src/filter_easy.py --out data/pass_rates_base.jsonl          # Qwen2.5-1.5B-Instruct
    python src/filter_easy.py --limit 50                                  # test rapide

Ensuite :
    python src/train_grpo.py --pass_rates data/pass_rates_sft.jsonl --max_pass 7 ...
"""
import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from train_grpo import build_dataset, correctness_reward, format_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=8, help="= num_generations de train_grpo.py")
    ap.add_argument("--temperature", type=float, default=1.0, help="= temperature de train_grpo.py")
    ap.add_argument("--max_tokens", type=int, default=512, help="= max_completion_length de train_grpo.py")
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/pass_rates.jsonl")
    args = ap.parse_args()

    train_ds, _ = build_dataset(args.n_val)          # exactement le train set de train_grpo.py
    if args.limit:
        train_ds = train_ds.select(range(args.limit))

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
               for ex in train_ds]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate(prompts, sp)

    records, hist = [], Counter()
    n_zero_std = 0
    for idx, (ex, out) in enumerate(zip(train_ds, outputs)):
        texts = [o.text for o in out.outputs]
        golds = [ex["gold"]] * len(texts)
        correct = correctness_reward(texts, golds)
        fmt = format_reward(texts)
        total = [c + f for c, f in zip(correct, fmt)]
        n_correct = int(sum(correct))
        zero_std = len(set(total)) == 1                 # meme critere que l'avantage GRPO nul
        hist[n_correct] += 1
        n_zero_std += zero_std
        records.append({
            "idx": idx,                                  # position dans le train set de train_grpo.py
            "question": ex["prompt"][0]["content"],
            "gold": ex["gold"],
            "n_correct": n_correct,
            "n": len(texts),
            "zero_std": zero_std,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_q = len(records)
    print("=" * 50)
    print(f"Modele                 : {args.model}")
    print(f"Questions              : {n_q} | tirages/question : {args.n} | T = {args.temperature}")
    print(f"Pass@1 moyen           : {100 * sum(r['n_correct'] for r in records) / (n_q * args.n):.2f} %")
    print(f"Groupes a variance nulle : {100 * n_zero_std / n_q:.1f} %")
    print("Repartition du nombre de bonnes reponses :")
    for k in range(args.n + 1):
        print(f"  {k}/{args.n} : {hist[k]:5d}  ({100 * hist[k] / n_q:5.1f} %)")
    print(f"Sortie                 : {args.out}")


if __name__ == "__main__":
    main()
