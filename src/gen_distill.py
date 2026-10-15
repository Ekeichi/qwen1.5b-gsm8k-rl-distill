"""
Generation des donnees de distillation : un professeur resout le train set GSM8K,
on ne garde que les solutions dont la reponse finale est correcte (rejection sampling).

Usage :
    python src/gen_distill.py                  # professeur Qwen2.5-7B-Instruct, 8 tirages par question
    python src/gen_distill.py --limit 50 --n 2 # test rapide

Sortie : data/distill_data.jsonl, une ligne par exemple retenu :
    {"messages": [user, assistant], "gold": "...", "qid": 123}

Le split de validation (300 premieres questions apres shuffle(seed=0)) est exclu, exactement
comme dans train_grpo.py, pour que la validation reste vierge.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from fractions import Fraction

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


# ---------------------------------------------------------------- parsing (meme logique que eval_gsm8k.py)
def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def to_number(s):
    if s is None:
        return None
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(",", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").strip()
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        try:
            val = Fraction(m.group(2)) / Fraction(m.group(3))
            return -val if m.group(1) else val
        except (ZeroDivisionError, ValueError):
            return None
    nums = NUM_RE.findall(s)
    if not nums:
        return None
    try:
        return Fraction(nums[0])
    except (ZeroDivisionError, ValueError):
        return None


def same(a, b):
    return a is not None and b is not None and abs(a - b) <= Fraction(1, 10000)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=8, help="tirages par question")
    ap.add_argument("--max_per_q", type=int, default=4, help="solutions correctes distinctes gardees par question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--max_keep_tokens", type=int, default=600, help="ecarte les solutions plus longues")
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/distill_data.jsonl")
    args = ap.parse_args()

    raw = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    raw = raw.select(range(args.n_val, len(raw)))          # on exclut la validation
    if args.limit:
        raw = raw.select(range(args.limit))

    tok = AutoTokenizer.from_pretrained(args.teacher)
    prompts_user = [f"{ex['question']}\n{INSTRUCTION}" for ex in raw]
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        for u in prompts_user
    ]
    golds = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in raw]

    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept = []
    n_samples = n_correct = covered = 0
    kept_tokens = 0
    for qid, (u, g, out) in enumerate(zip(prompts_user, golds, outputs)):
        gold = to_number(g)
        good, seen = [], set()
        for o in out.outputs:
            n_samples += 1
            text = o.text.strip()
            ok = (o.finish_reason == "stop"                      # pas de sortie tronquee
                  and text.count("\\boxed") == 1                  # un seul \boxed, en fin de raisonnement
                  and same(to_number(last_boxed(text)), gold))
            if ok:
                n_correct += 1
                if len(o.token_ids) <= args.max_keep_tokens and text not in seen:
                    seen.add(text)
                    good.append((text, len(o.token_ids)))
        if good:
            covered += 1
        rng.shuffle(good)                                        # diversite, pas de biais vers la 1ere
        for text, n_tok in good[:args.max_per_q]:
            kept_tokens += n_tok
            kept.append({
                "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": text}],
                "gold": g,
                "qid": qid,
            })

    rng.shuffle(kept)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_q = len(golds)
    print("=" * 50)
    print(f"Professeur              : {args.teacher}")
    print(f"Questions               : {n_q} | tirages/question : {args.n}")
    print(f"Accuracy prof (pass@1)  : {100 * n_correct / n_samples:.2f} %")
    print(f"Couverture (>=1 correct): {100 * covered / n_q:.2f} %  ({covered}/{n_q})")
    print(f"Exemples gardes         : {len(kept)}  (moyenne {len(kept) / n_q:.2f} par question)")
    print(f"Tokens moyens gardes    : {kept_tokens / max(1, len(kept)):.0f}")
    print(f"Sortie                  : {args.out}")


if __name__ == "__main__":
    main()
