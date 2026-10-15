#!/usr/bin/env bash
# Version pour la reprise : evalue ckpt300, ckpt400 et final directement depuis
# outputs/grpo_gsm8k (pas de copie, pour economiser le disque).
# ckpt100 et ckpt200 ont deja leurs resultats (fichiers *_greedy.jsonl et *_maj16.jsonl).
#
# A lancer APRES avoir demarre l'entrainement (il detecte la fin de l'entrainement) :
#   nohup bash scripts/watch_eval_resume.sh > logs/watch_eval_resume.log 2>&1 &

cd ~/itw || exit 1
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

EVAL=src/eval_gsm8k.py
OUT=outputs/grpo_gsm8k
mkdir -p logs/eval results/eval

training_running() { pgrep -f train_grpo.py > /dev/null; }

wait_for() {
  while [ ! -f "$1" ]; do
    if ! training_running; then
      [ -f "$1" ] && break
      return 1
    fi
    sleep 15
  done
  sleep 30
  return 0
}

run_eval() {
  name=$1
  model=$2
  mem=0.10
  training_running || mem=0.40
  echo "[$(date +%H:%M)] eval $name (gpu_mem=$mem)"
  [ -f results/eval/${name}_greedy.jsonl ] || python $EVAL --model $model --gpu_mem $mem \
      --out results/eval/${name}_greedy.jsonl > logs/eval/${name}_greedy.log 2>&1
  [ -f results/eval/${name}_maj16.jsonl ] || python $EVAL --model $model --gpu_mem $mem \
      --n 16 --temperature 0.7 --top_p 0.95 \
      --out results/eval/${name}_maj16.jsonl > logs/eval/${name}_maj16.log 2>&1
  python src/compare.py | tee results/summary.txt
}

if ! training_running; then
  echo "L'entrainement ne tourne pas : lance src/train_grpo.py d'abord."
  exit 1
fi

for s in 300 400; do
  if wait_for $OUT/checkpoint-$s/model.safetensors; then
    run_eval ckpt$s ~/itw/$OUT/checkpoint-$s
  else
    echo "[$(date +%H:%M)] checkpoint-$s introuvable, abandon"
  fi
done

if wait_for $OUT/final/model.safetensors; then
  while training_running; do sleep 10; done
  run_eval final ~/itw/$OUT/final
fi

echo "[$(date +%H:%M)] termine"
