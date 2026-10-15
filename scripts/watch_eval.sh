#!/usr/bin/env bash
# Surveille l'entrainement GRPO, copie chaque checkpoint avant la rotation,
# l'evalue (greedy + 16 tirages) puis met a jour summary.txt.
#
# Lancement (sur le serveur, depuis ~/itw), une seule fois :
#   nohup bash scripts/watch_eval.sh > logs/watch_eval.log 2>&1 &
# Suivi :
#   tail -f logs/watch_eval.log    cat results/summary.txt    ls logs/eval/

cd ~/itw || exit 1
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

EVAL=src/eval_gsm8k.py            # script d'evaluation (celui que tu as utilise jusqu'ici)
OUT=outputs/grpo_gsm8k
mkdir -p logs/eval results/eval

training_running() { pgrep -f train_grpo.py > /dev/null; }

# Attend qu'un fichier existe. Abandonne si l'entrainement est fini et que le fichier n'est pas la.
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

# Copie chaque checkpoint des qu'il est ecrit (save_total_limit=3 supprime les plus anciens).
copier() {
  for s in 100 200 300 400; do
    if [ -f ckpt$s/model.safetensors ]; then
      touch ckpt$s/.copied
      continue
    fi
    if wait_for $OUT/checkpoint-$s/model.safetensors; then
      cp -r $OUT/checkpoint-$s ckpt$s && touch ckpt$s/.copied
      echo "[$(date +%H:%M)] ckpt$s copie"
    else
      echo "[$(date +%H:%M)] checkpoint-$s introuvable, abandon"
    fi
  done
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

copier &
COPIER_PID=$!

for s in 100 200 300 400; do
  while [ ! -f ckpt$s/.copied ] && kill -0 $COPIER_PID 2> /dev/null; do sleep 10; done
  if [ -f ckpt$s/.copied ]; then
    run_eval ckpt$s ~/itw/ckpt$s
  fi
done

# Modele final : on attend la fin complete de l'entrainement pour pouvoir donner plus de memoire a vLLM.
if wait_for $OUT/final/model.safetensors; then
  while training_running; do sleep 10; done
  run_eval final ~/itw/$OUT/final
fi

echo "[$(date +%H:%M)] termine"
