"""
SFT du 1.5B sur les solutions du professeur (sortie de gen_distill.py).
Boucle PyTorch simple, volontairement sans TRL (son API change souvent) :
perte uniquement sur la reponse de l'assistant, poids maitres en fp32, calcul en bf16.

Usage :
    python src/sft_distill.py                                   # depart Qwen2.5-1.5B-Instruct
    python src/sft_distill.py --max_steps 5 --out outputs/sft_test   # test rapide

Ensuite :
    python src/eval_gsm8k.py --model outputs/sft_distill --out results/eval/sft_greedy.jsonl
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_example(tok, messages, max_len):
    """(input_ids, labels) avec labels = -100 sur le prompt. None si inutilisable."""
    prompt_text = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, tokenize=False).rstrip("\n")  # finit sur <|im_end|>
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def collate(examples, pad_id, device):
    L = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), L), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1
    return input_ids.to(device), labels.to(device), mask.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/distill_data.jsonl")
    ap.add_argument("--out", default="outputs/sft_distill")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4, help="sequences par micro-batch")
    ap.add_argument("--grad_accum", type=int, default=16, help="batch effectif = batch x grad_accum")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=-1, help="test rapide, ex: 5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dtype", default="bf16", choices=["bf16", "fp32"],
                    help="bf16 = ~3 Go sur disque (defaut), fp32 = ~6 Go")
    ap.add_argument("--max_skips", type=int, default=20,
                    help="nb max de pas ignores (gradient non fini) avant d'arreter")
    ap.add_argument("--no_save", action="store_true", help="ne pas sauvegarder le modele (tests)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="implementation de l'attention (eager = plus lent, utile pour diagnostiquer un NaN)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    data = [e for e in (build_example(tok, r["messages"], args.max_len) for r in rows) if e is not None]
    print(f"{len(data)} exemples utilisables sur {len(rows)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation=args.attn).to(device)
    model.gradient_checkpointing_enable()
    # controle : doit afficher float32 (sinon l'argument dtype a ete ignore par cette version de transformers)
    print(f"dtype des poids : {next(model.parameters()).dtype} | attention : {args.attn}", flush=True)
    model.config.use_cache = False
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    per_step = args.batch * args.grad_accum
    steps_per_epoch = math.ceil(len(data) / per_step)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    def lr_at(step):                     # warmup lineaire puis cosinus jusqu'a 10 % du lr
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"{total_steps} steps, batch effectif {per_step}, lr {args.lr}")

    def localize(chunk_idx):
        """Rejoue le lot fautif micro-batch par micro-batch, puis exemple par exemple."""
        def probe(idx):
            ids, labels, mask = collate([data[i] for i in idx], pad_id, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            n = max(1, (labels[:, 1:] != -100).sum().item())
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / n
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            ok = bool(torch.isfinite(loss) and torch.isfinite(g))
            opt.zero_grad(set_to_none=True)
            return ok, loss.item(), float(g)

        print("localisation du lot fautif...", flush=True)
        for m in range(0, len(chunk_idx), args.batch):
            idx = chunk_idx[m:m + args.batch]
            ok, l, g = probe(idx)
            if ok:
                continue
            print(f"  micro-batch {m // args.batch} fautif : longueurs {[len(data[i][0]) for i in idx]}, "
                  f"loss {l}, grad_norm {g}", flush=True)
            singles = [(i, *probe([i])) for i in idx]
            bad = [(i, l1, g1) for i, ok1, l1, g1 in singles if not ok1]
            for i, l1, g1 in bad:
                print(f"    exemple seul fautif : indice {i}, longueur {len(data[i][0])}, "
                      f"tokens de perte {sum(1 for x in data[i][1] if x != -100)}, loss {l1}, grad_norm {g1}", flush=True)
                print(f"    debut : {tok.decode(data[i][0][:60])!r}", flush=True)
            if not bad:
                print("    chaque exemple passe seul : probleme lie au padding ou au melange du lot", flush=True)
        opt.zero_grad(set_to_none=True)

    step, t0, run_loss, run_n, skipped = 0, time.time(), 0.0, 0, 0
    while step < total_steps:
        order = list(range(len(data)))
        random.shuffle(order)
        for s in range(steps_per_epoch):
            if step >= total_steps:
                break
            chunk = order[s * per_step:(s + 1) * per_step]
            if not chunk:
                continue
            n_tok = sum(sum(1 for x in data[i][1] if x != -100) for i in chunk)   # normalisation par token
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step_loss = 0.0
            for m in range(0, len(chunk), args.batch):
                ids, labels, mask = collate([data[i] for i in chunk[m:m + args.batch]], pad_id, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum") / n_tok
                if not torch.isfinite(loss):
                    lg = logits.detach().float()
                    print(f"NaN/inf au step {step + 1}, micro-batch {m // args.batch}: "
                          f"forme {tuple(ids.shape)}, tokens de perte {(labels != -100).sum().item()}, "
                          f"logits finis: {torch.isfinite(lg).all().item()}, "
                          f"max|logits| {lg.nan_to_num(0.0, 0.0, 0.0).abs().max().item():.1f}", flush=True)
                    raise SystemExit(1)
                loss.backward()
                step_loss += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(gn)):
                # Gradient non fini : on NE fait PAS le pas d'optimisation, il corromprait les poids
                # (c'est ce qui a produit la loss NaN du premier essai).
                skipped += 1
                print(f"grad_norm non fini au step {step + 1} (loss {step_loss:.4f}) : pas ignore "
                      f"({skipped}/{args.max_skips})", flush=True)
                opt.zero_grad(set_to_none=True)
                if skipped == 1:
                    localize(chunk)
                if skipped >= args.max_skips:
                    print("trop de pas ignores : arret", flush=True)
                    raise SystemExit(1)
                step += 1
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += step_loss
            run_n += 1
            if step <= 3 or step % 10 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} | loss {run_loss / run_n:.4f} | grad_norm {gn:.2f} "
                      f"| lr {lr_at(step - 1):.2e} | {time.time() - t0:.0f}s", flush=True)
                run_loss, run_n = 0.0, 0

    print(f"termine : {step} steps, {skipped} pas ignores", flush=True)
    if args.no_save:
        print("--no_save : pas de sauvegarde")
        return
    model.config.use_cache = True
    if args.save_dtype == "bf16":
        model = model.to(torch.bfloat16)  # 3 Go sur disque au lieu de 6 ; vLLM travaille en bf16 de toute facon
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"modele sauvegarde dans {args.out}")


if __name__ == "__main__":
    main()
