#!/usr/bin/env bash
# Evalue chaque checkpoint d'un run GRPO directement dans son dossier de sortie (pas de copie,
# pour economiser le disque), puis le modele final, et met a jour results/summary.txt.
# Avec --save_total_limit 1, un checkpoint est supprime ~25 min apres son ecriture :
# l'eval (greedy + 16 tirages, ~10 min) a le temps de passer.
#
# Usage (depuis la racine du depot, APRES avoir lance l'entrainement) :
#   nohup bash scripts/watch_eval_run.sh outputs/grpo_sft sft_grpo > logs/watch_eval_sft_grpo.log 2>&1 &
# -> results/eval/sft_grpo100_greedy.jsonl, ..., results/eval/sft_grpo_final_maj16.jsonl

OUT=${1:?usage: watch_eval_run.sh <dossier de sortie du run> <prefixe des resultats>}
PREFIX=${2:?usage: watch_eval_run.sh <dossier de sortie du run> <prefixe des resultats>}

cd ~/itw || exit 1
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

EVAL=src/eval_gsm8k.py
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
  sleep 30   # laisse finir l'ecriture des fichiers
  return 0
}

run_eval() {
  name=$1
  model=$2
  mem=0.10                                 # petit pour ne pas faire sauter l'entrainement
  training_running || mem=0.40             # plus rapide une fois l'entrainement termine
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

for s in 100 200 300 400; do
  if wait_for $OUT/checkpoint-$s/model.safetensors; then
    run_eval ${PREFIX}$s ~/itw/$OUT/checkpoint-$s
  else
    echo "[$(date +%H:%M)] checkpoint-$s introuvable, abandon"
  fi
done

if wait_for $OUT/final/model.safetensors; then
  while training_running; do sleep 10; done
  run_eval ${PREFIX}_final ~/itw/$OUT/final
fi

echo "[$(date +%H:%M)] termine"
