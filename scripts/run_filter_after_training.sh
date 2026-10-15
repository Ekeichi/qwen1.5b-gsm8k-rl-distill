#!/usr/bin/env bash
# Attend la fin du run GRPO en cours et de ses evaluations, puis calcule les taux de reussite
# par question (filter_easy.py) pour le modele SFT et le modele de base.
#   nohup bash scripts/run_filter_after_training.sh > logs/filter_easy.log 2>&1 &
cd ~/itw || exit 1
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

while pgrep -f "src/train_grpo.py|watch_eval_run.sh|src/eval_gsm8k.py" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] GPU libre, debut du filtrage"

[ -f data/pass_rates_sft.jsonl ] || python src/filter_easy.py --model outputs/sft_distill --out data/pass_rates_sft.jsonl
[ -f data/pass_rates_base.jsonl ] || python src/filter_easy.py --out data/pass_rates_base.jsonl
echo "[$(date +%H:%M)] termine"
