# GSM8K : GRPO vs distillation sur Qwen2.5-1.5B-Instruct

Deux façons d'améliorer un petit modèle (Qwen2.5-1.5B-Instruct) en raisonnement mathématique sur GSM8K :

1. **GRPO** (apprentissage par renforcement, TRL) avec une récompense binaire « bonne réponse dans `\boxed{}` ».
2. **Distillation** : SFT sur des solutions correctes générées par un professeur (Qwen2.5-7B-Instruct, rejection sampling).

Tout a été entraîné et évalué sur une seule H100 80 Go.

## Résultats (GSM8K test, 1319 questions)

| modèle | greedy | pass@1 | maj@16 | gain pass@1 vs baseline (IC 95 %) |
|---|---:|---:|---:|---|
| baseline (Qwen2.5-1.5B-Instruct) | 74.22 | 73.15 | 83.85 | – |
| GRPO step 100 | 75.28 | 74.05 | 82.34 | +0.90 (± 0.70) |
| GRPO step 200 | 75.06 | 75.32 | 82.49 | +2.17 (± 0.81) |
| GRPO step 300 | 77.48 | 76.73 | 83.62 | +3.58 (± 0.86) |
| GRPO step 400 | 77.79 | 77.26 | 83.40 | +4.11 (± 0.89) |
| **GRPO final (448 steps)** | 77.63 | **77.55** | 83.70 | **+4.40 (± 0.95)** |
| SFT distillation | 77.48 | 76.93 | **84.99** | +3.78 (± 0.81) |

- **greedy** : 1 génération à température 0.
- **pass@1** : précision moyenne sur 16 tirages (T = 0.7, top_p = 0.95).
- **maj@16** : vote majoritaire sur ces 16 tirages.
- **IC 95 %** : différence appariée question par question (pass@1 sur 16 tirages) ; un gain est significatif si l'intervalle n'inclut pas 0. Le greedy vLLM varie d'environ ± 0.2 pt d'un run à l'autre.

**À retenir.** Les deux méthodes gagnent environ +4 pts en pass@1. GRPO améliore la réponse « du premier coup » mais pas le maj@16 (probablement parce qu'il réduit la diversité des tirages : l'entropie tombe à ~0.07 en fin d'entraînement) ; la distillation donne le meilleur maj@16 (+1.1 pt, écart non testé statistiquement).

Tableau brut : [`results/summary.txt`](results/summary.txt), régénérable avec `python src/compare.py`.

## Organisation du dépôt

```
src/
  eval_gsm8k.py     évaluation vLLM (greedy ou n tirages + vote majoritaire)
  train_grpo.py     GRPO avec TRL (vLLM colocalisé pour la génération)
  gen_distill.py    génération des données de distillation par le professeur 7B
  sft_distill.py    SFT du 1.5B sur ces données (boucle PyTorch simple)
  compare.py        tableau récapitulatif + intervalles de confiance
scripts/
  watch_eval.sh         évalue chaque checkpoint GRPO pendant l'entraînement
  watch_eval_resume.sh  même chose pour la reprise (checkpoints 300, 400, final)
  watch_eval_run.sh     version générique : watch_eval_run.sh <dossier du run> <préfixe des résultats>
data/
  val_gsm8k.jsonl               300 questions de validation (tirées du train, exclues de l'entraînement)
  distill_data.jsonl            27 799 solutions correctes du professeur
  distill_data_smoketest.jsonl  petit échantillon du test rapide de gen_distill.py
results/
  summary.txt       tableau ci-dessus
  eval/<modèle>_greedy.jsonl, <modèle>_maj16.jsonl   prédictions détaillées par question
  eval/baseline_greedy_first100.jsonl                premier essai (100 questions seulement)
logs/
  train_grpo_run1.log, train_grpo_run2_resume.log, sft_distill.log, ...
  eval/             logs de chaque évaluation
  tensorboard/      courbes GRPO (récompense, longueur, KL...)
outputs/            poids des modèles (non versionnés, voir .gitignore)
```

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export VLLM_USE_FLASHINFER_SAMPLER=0   # voir « Notes techniques »
```

Toutes les commandes se lancent **depuis la racine du dépôt**.

## Reproduire

```bash
# 1. Baseline
python src/eval_gsm8k.py --out results/eval/baseline_greedy.jsonl
python src/eval_gsm8k.py --n 16 --temperature 0.7 --top_p 0.95 --out results/eval/baseline_maj16.jsonl

# 2. GRPO (~2 h sur H100) ; checkpoints tous les 100 steps dans outputs/grpo_gsm8k
python src/train_grpo.py --report_to tensorboard
#    en parallèle, évaluation automatique des checkpoints :
nohup bash scripts/watch_eval.sh > logs/watch_eval.log 2>&1 &

# 3. Distillation
python src/gen_distill.py                       # professeur Qwen2.5-7B-Instruct, 8 tirages/question
python src/sft_distill.py                       # 870 steps, ~45 min
python src/eval_gsm8k.py --model outputs/sft_distill --out results/eval/sft_greedy.jsonl
python src/eval_gsm8k.py --model outputs/sft_distill --n 16 --temperature 0.7 --top_p 0.95 \
    --out results/eval/sft_maj16.jsonl

# 4. GRPO à partir du modèle distillé (SFT puis GRPO)
VLLM_USE_FLASHINFER_SAMPLER=0 nohup python src/train_grpo.py --model outputs/sft_distill \
    --out outputs/grpo_sft --logging_dir logs/tensorboard/grpo_sft \
    --save_total_limit 1 --report_to tensorboard > logs/train_grpo_sft.log 2>&1 &
nohup bash scripts/watch_eval_run.sh outputs/grpo_sft sft_grpo > logs/watch_eval_sft_grpo.log 2>&1 &

# 5. Tableau
python src/compare.py | tee results/summary.txt
```

`train_grpo.py` refuse de démarrer si `<out>/final` existe déjà (protège les modèles déjà entraînés).

Courbes d'entraînement : `tensorboard --logdir logs/tensorboard`.

## Hyperparamètres principaux

| | GRPO | SFT distillation |
|---|---|---|
| départ | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| données | GSM8K train (7 173 questions, 300 en validation) | 27 799 solutions du 7B (≤ 4 par question, ≤ 600 tokens) |
| lr | 1e-6 constant (warmup 3 %) | 1e-5, warmup 20 steps puis cosinus → 10 % |
| batch | 32 complétions × 8 accum. (8 générations par prompt) | 4 × 16 accum. = 64 séquences |
| durée | 2 epochs = 448 steps | 2 epochs = 870 steps |
| autres | β (KL) = 0, T = 1.0, 512 tokens max | perte sur la réponse uniquement, poids fp32 / calcul bf16 |

Récompense GRPO : 1.0 si le dernier `\boxed{}` vaut la bonne réponse, + 0.1 si un `\boxed{}` non vide est présent.

## Déroulé réel des expériences

- **GRPO run 1** (`logs/train_grpo_run1.log`) : s'est arrêté au step 300, disque plein lors de la sauvegarde. Les checkpoints 100 et 200 ont été évalués.
- **GRPO run 2** (`logs/train_grpo_run2_resume.log`) : reprise depuis `checkpoint-200`. Ce checkpoint avait été sauvé avec `save_only_model` (sans état d'optimiseur) : la reprise a gardé les poids et le compteur de steps, mais **Adam et le scheduler sont repartis de zéro**. Les résultats step 300, 400 et final viennent de ce run.
- **SFT** : un test de 5 steps (`logs/sft_distill_smoketest.log`) puis l'entraînement complet (`logs/sft_distill.log`, 0 pas ignoré).

## Notes techniques

- **`VLLM_USE_FLASHINFER_SAMPLER=0`** : sans CUDA toolkit (`nvcc`) sur la machine, le sampler FlashInfer ne peut pas compiler ses kernels ; on utilise le sampler PyTorch de vLLM.
- **transformers ≥ 5** : `warmup_ratio` n'existe plus, on passe `warmup_steps=0.03` (un float < 1 est lu comme un ratio). Le dossier TensorBoard se règle via la variable `TENSORBOARD_LOGGING_DIR` (fait dans `train_grpo.py`).
- **Disque** : un checkpoint complet fait 18 Go (6 Go de poids fp32 + 12 Go d'état Adam). `--save_only_model 1` (défaut) le ramène à 6 Go, mais le checkpoint n'est plus reprenable à l'identique.
- **Mémoire GPU** : l'évaluation pendant l'entraînement GRPO tourne avec `--gpu_mem 0.10` pour cohabiter avec le vLLM colocalisé.
