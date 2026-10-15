# GSM8K : GRPO vs distillation sur Qwen2.5-1.5B-Instruct

Deux façons d'améliorer un petit modèle (Qwen2.5-1.5B-Instruct) en raisonnement mathématique sur GSM8K :

1. **GRPO** (apprentissage par renforcement, TRL) avec une récompense binaire « bonne réponse dans `\boxed{}` ».
2. **Distillation** : SFT sur des solutions correctes générées par un professeur (Qwen2.5-7B-Instruct, rejection sampling).
3. **Distillation puis GRPO** : le même GRPO, en partant du modèle distillé.

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
| SFT distillation | 77.48 | 76.93 | 84.99 | +3.78 (± 0.81) |
| SFT + GRPO step 100 | 77.71 | 77.40 | 85.06 | +4.25 (± 0.86) |
| SFT + GRPO step 200 | 78.39 | 77.98 | **85.22** | +4.82 (± 0.92) |
| SFT + GRPO step 300 | **78.92** | 77.97 | 84.23 | +4.82 (± 0.94) |
| SFT + GRPO step 400 | 78.01 | 78.14 | 83.47 | +4.98 (± 0.96) |
| **SFT + GRPO final (448 steps)** | 78.77 | **78.63** | 83.85 | **+5.48 (± 1.00)** |

- **greedy** : 1 génération à température 0.
- **pass@1** : précision moyenne sur 16 tirages (T = 0.7, top_p = 0.95).
- **maj@16** : vote majoritaire sur ces 16 tirages.
- **IC 95 %** : différence appariée question par question (pass@1 sur 16 tirages) ; un gain est significatif si l'intervalle n'inclut pas 0. Le greedy vLLM varie d'environ ± 0.2 pt d'un run à l'autre.

Comparaisons appariées (question par question) du modèle SFT + GRPO final :

| comparé à | pass@1 | maj@16 (corrigées / cassées) |
|---|---|---|
| SFT | **+1.70 (± 0.79)** | 38 / 53 (McNemar p = 0.14) |
| GRPO seul, final | +1.08 (± 1.10) | 61 / 59 (p = 0.93) |

**À retenir.**
- **Distillation puis GRPO est la meilleure chaîne** : 78.63 en pass@1, +5.5 pts sur la baseline. Le GRPO ajoute +1.7 pt (significatif) au modèle distillé ; l'écart avec le GRPO seul (+1.1 pt) est à la limite de la significativité.
- GRPO seul et distillation seule gagnent chacun environ +4 pts en pass@1.
- **GRPO n'améliore pas le maj@16** : le GRPO seul le laisse au niveau de la baseline, et sur le modèle distillé il monte jusqu'au step 200 (85.22) puis redescend (83.47 au step 400, dégradation significative vs SFT, p = 0.03). C'est cohérent avec l'effondrement de l'entropie (voir Diagnostic) : GRPO rend chaque tirage plus fiable mais moins divers. La distillation donne le meilleur maj@16.
- En pratique : le modèle final pour une réponse unique, le step 200 pour du vote majoritaire (ses poids n'ont pas été conservés, `--save_total_limit 1`).

Tableau brut : [`results/summary.txt`](results/summary.txt), régénérable avec `python src/compare.py`.

## Organisation du dépôt

```
src/
  eval_gsm8k.py     évaluation vLLM (greedy ou n tirages + vote majoritaire)
  train_grpo.py     GRPO avec TRL (vLLM colocalisé pour la génération)
  gen_distill.py    génération des données de distillation par le professeur 7B
  sft_distill.py    SFT du 1.5B sur ces données (boucle PyTorch simple)
  compare.py        tableau récapitulatif + intervalles de confiance
  filter_easy.py    taux de réussite par question du train (pour filtrer les questions sans signal GRPO)
scripts/
  watch_eval.sh         évalue chaque checkpoint GRPO pendant l'entraînement
  watch_eval_resume.sh  même chose pour la reprise (checkpoints 300, 400, final)
  watch_eval_run.sh     version générique : watch_eval_run.sh <dossier du run> <préfixe des résultats>
  run_filter_after_training.sh  lance filter_easy.py (SFT puis base) dès que le GPU est libre
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
- **SFT + GRPO** (`logs/train_grpo_sft.log`, 15/10, 448 steps en 2 h 02, sans incident) : mêmes hyperparamètres que le GRPO seul, départ `outputs/sft_distill`, `--save_total_limit 1` (disque partagé). Checkpoints évalués au fil de l'eau par `scripts/watch_eval_run.sh` (résultats `results/eval/sft_grpo*`, courbes `logs/tensorboard/grpo_sft`).

## Diagnostic : pourquoi GRPO plafonne

Mesures tirées de TensorBoard (moyennes début = steps 5-25, fin = steps 408-448) :

| | GRPO seul, début | GRPO seul, fin | SFT + GRPO, début |
|---|---:|---:|---:|
| récompense (bonnes réponses, T = 1.0) | 0.745 | 0.887 | 0.889 |
| `frac_reward_zero_std` | 0.35 | **0.73** | **0.70** |
| entropie | 0.25 | 0.07 | 0.11 |
| longueur moyenne (tokens) | 288 | 259 | 286 |
| sorties tronquées (512 tokens) | 4.0 % | 0.9 % | 2.3 % |

- **Ce n'est pas la longueur** : les sorties raccourcissent, presque aucune n'est tronquée (0.23 % en éval à 1024 tokens), et les données de distillation sont du CoT standard (0 balise `<think>`, « wait » dans 0.4 % des exemples).
- **C'est le manque de signal** : `frac_reward_zero_std` est la part de questions dont les 8 réponses ont la même récompense. Leur avantage vaut exactement 0 (`grpo_trainer.py:2711-2713`, TRL 1.15) : elles coûtent génération + forward/backward pour un gradient nul. En fin de GRPO, 3 questions sur 4 ; le modèle SFT démarre directement là.
- **Et ces groupes diluent le gradient des autres** : la loss par défaut (`loss_type="dapo"`) divise par le nombre total de tokens du batch, groupes nuls compris (`grpo_trainer.py:2424`, `3094-3099`). Avec 73 % de groupes nuls, le lr effectif tombe à ~0.27 × 1e-6.
- L'entropie s'effondre et le maj@16 ne bouge pas : GRPO rend le modèle plus fiable sur des réponses qu'il savait déjà produire, plus qu'il n'ajoute de capacité.

## Pistes pour la suite (non faites)

1. **Filtrer hors ligne les questions trop faciles** (outillage prêt, pas encore entraîné) :
   ```bash
   # taux de réussite par question, mêmes conditions que l'entraînement (8 tirages, T = 1.0, 512 tokens) ~10 min
   python src/filter_easy.py --model outputs/sft_distill --out data/pass_rates_sft.jsonl
   # GRPO sur les seules questions où le modèle n'a pas 8/8
   VLLM_USE_FLASHINFER_SAMPLER=0 python src/train_grpo.py --model outputs/sft_distill \
       --pass_rates data/pass_rates_sft.jsonl --max_pass 7 \
       --out outputs/grpo_sft_filtered --logging_dir logs/tensorboard/grpo_sft_filtered \
       --save_total_limit 1 --report_to tensorboard
   ```
   `--min_pass 1` retire aussi les questions jamais réussies (0/8, également sans signal). Moins de questions = moins de steps par epoch : augmenter `--epochs` pour un budget comparable. Le filtre vieillit pendant l'entraînement (une question à 7/8 peut passer à 8/8) : on peut le recalculer entre deux epochs.
   Les taux de réussite du modèle SFT et du modèle de base sont en cours de calcul sur le serveur (`data/pass_rates_sft.jsonl`, `data/pass_rates_base.jsonl`, log `logs/filter_easy.log`), pas encore versionnés.
2. **Plus de diversité** : 16 générations par prompt au lieu de 8, température plus haute, ou β > 0 (KL) pour freiner l'effondrement d'entropie.
3. **Corriger la dilution** dans la loss (normaliser seulement sur les groupes à std > 0, en sous-classant `compute_loss`) ou implémenter le *dynamic sampling* de DAPO (absent de TRL 1.15).
4. **Données plus dures** (MATH, variantes de GSM8K) : c'est là que du raisonnement long (`<think>`) aurait un intérêt.

## Notes techniques

- **`VLLM_USE_FLASHINFER_SAMPLER=0`** : sans CUDA toolkit (`nvcc`) sur la machine, le sampler FlashInfer ne peut pas compiler ses kernels ; on utilise le sampler PyTorch de vLLM.
- **transformers ≥ 5** : `warmup_ratio` n'existe plus, on passe `warmup_steps=0.03` (un float < 1 est lu comme un ratio). Le dossier TensorBoard se règle via la variable `TENSORBOARD_LOGGING_DIR` (fait dans `train_grpo.py`).
- **Disque** : un checkpoint complet fait 18 Go (6 Go de poids fp32 + 12 Go d'état Adam). `--save_only_model 1` (défaut) le ramène à 6 Go, mais le checkpoint n'est plus reprenable à l'identique.
- **Mémoire GPU** : l'évaluation pendant l'entraînement GRPO tourne avec `--gpu_mem 0.10` pour cohabiter avec le vLLM colocalisé.
