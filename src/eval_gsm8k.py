"""
Evaluation GSM8K avec vLLM (modele de base, checkpoint GRPO ou modele SFT).

Usage :
    pip install vllm datasets transformers
    python src/eval_gsm8k.py --out results/eval/baseline_greedy.jsonl          # greedy, Qwen2.5-1.5B-Instruct
    python src/eval_gsm8k.py --n 16 --temperature 0.7 --top_p 0.95 --out results/eval/baseline_maj16.jsonl  # pass@1 + maj@16
    python src/eval_gsm8k.py --model outputs/grpo_gsm8k/final --out results/eval/final_greedy.jsonl
    python src/eval_gsm8k.py --limit 100       # test rapide
"""
import argparse
import json
import os
import re
from collections import Counter
from fractions import Fraction

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------- parsing
def last_boxed(text):
    """Contenu du dernier \\boxed{...} (avec gestion des accolades imbriquees)."""
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
    return None  # accolade jamais fermee (souvent une sortie tronquee)


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def to_number(s, take="first"):
    """Convertit une chaine en Fraction. Retourne None si aucun nombre."""
    if s is None:
        return None
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(",", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.strip()

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
    tok = nums[0] if take == "first" else nums[-1]
    try:
        return Fraction(tok)
    except (ZeroDivisionError, ValueError):
        return None


def extract_pred(text):
    """Reponse predite : \\boxed en priorite, sinon dernier nombre du texte."""
    boxed = last_boxed(text)
    if boxed is not None:
        val = to_number(boxed, take="first")
        if val is not None:
            return val, True
    return to_number(text, take="last"), False


def gold_answer(answer_field):
    return to_number(answer_field.split("####")[-1], take="first")


def same(a, b):
    """Comparaison exacte en Fractions (pas de float : un nombre geant ferait planter float())."""
    if a is None or b is None:
        return False
    return abs(a - b) <= Fraction(1, 10000)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=1, help="nb d'echantillons par question")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--out", default="results/eval/gsm8k_results.jsonl")
    args = ap.parse_args()

    if args.n > 1 and args.temperature == 0.0:
        raise SystemExit("n > 1 demande une temperature > 0 (ex: --temperature 0.7)")

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{ex['question']}\n{INSTRUCTION}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds
    ]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate(prompts, sp)

    n_samples = n_correct = n_boxed = n_trunc = total_tokens = n_maj_correct = 0
    records = []
    for ex, out in zip(ds, outputs):
        gold = gold_answer(ex["answer"])
        preds, flags = [], []
        for o in out.outputs:
            pred, boxed = extract_pred(o.text)
            ok = same(pred, gold)
            preds.append(pred)
            flags.append(ok)
            n_samples += 1
            n_correct += ok
            n_boxed += boxed
            n_trunc += (o.finish_reason == "length")
            total_tokens += len(o.token_ids)

        # vote majoritaire (utile seulement si n > 1)
        votes = Counter(p for p in preds if p is not None)  # Fractions : hashables et exactes
        maj = votes.most_common(1)[0][0] if votes else None
        maj_ok = same(maj, gold)
        n_maj_correct += maj_ok

        records.append({
            "question": ex["question"],
            "gold": str(gold),
            "preds": [None if p is None else str(p) for p in preds],
            "correct": flags,
            "maj": None if maj is None else str(maj),
            "maj_correct": maj_ok,
            "first_output": out.outputs[0].text,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_q = len(ds)
    print("=" * 50)
    print(f"Modele            : {args.model}")
    print(f"Questions         : {n_q} | echantillons/question : {args.n}")
    print(f"Accuracy (pass@1) : {100 * n_correct / n_samples:.2f} %")
    if args.n > 1:
        print(f"Accuracy maj@{args.n:<3}    : {100 * n_maj_correct / n_q:.2f} %")
    print(f"Taux de \\boxed    : {100 * n_boxed / n_samples:.2f} %")
    print(f"Sorties tronquees : {100 * n_trunc / n_samples:.2f} %")
    print(f"Tokens moyens     : {total_tokens / n_samples:.0f}")
    print(f"Details           : {args.out}")


if __name__ == "__main__":
    main()
