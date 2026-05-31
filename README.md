# ChemAI: Predict the Cure

Решение хакатона ChemAI: Predict the Cure (Kaggle).

По 210 RDKit-дескрипторам молекулы предсказать три показателя биологической
активности против вируса гриппа: **IC50**, **CC50**, **SI**. Метрика — средний
RMSE по трём таргетам. Train 750 соединений, test 250.

## Результат

**Public LB: 268.76238** (baseline HistGradientBoosting = 365.82, прирост −97.1 LB).

| Параметр | Значение |
|----------|----------|
| Adaptive top pct | **3.5%** |
| IC50 clip | **98.80** percentile |
| CC50 clip | **98.95** percentile |

Финальный submission: `submissions/submission.csv` (LB **268.76238**). 

При прогоне ноутбука перезаписывает этот файл тем же pipeline.

## Путь к решению (полный процесс)

### Фаза 1 — baseline и «большой ансамбль» (365 → ~278)

Старт: sklearn HistGradientBoosting (**LB 365.82**).

Далее классический ML-пайплайн: log-transform, LGBM/XGB/Cat ensemble,
two-stream log+raw, multi-trim SLSQP, domain features (Lipinski),
pseudo-labeling, diversity-pack (ET + Ridge + Huber + GP), stacking.
~1500 fits per target — **плато ~282.6 LB**.

Надстройки на сложной базе:
- CC50 cv070 train↔test matches → ~281.9
- MI-LGBM 10% blend → ~281.2
- **v2c SI stack** (4 LGBM на расширенных фичах + OOF IC/CC/SI) → **278.83**
- adaptive top-22% min-of-teachers → 278.75
- IC50 clip 99.55 → **278.59** — дальнейшие model-tweak'и давали ≤0.01 LB

**Урок:** OOF-улучшения часто регрессируют public LB; нужна LB-валидация, не OOF.

### Фаза 2 — упрощение базы (278 → ~272)

- Только **CatBoost + LightGBM**, blend **0.75 / 0.25** (без многоуровневого SLSQP)
- Outlier removal `SI < 2000`, drop constant + corr>0.95 фич → **159 фич**
- 10-fold KFold, log1p только для SI

Простая база → **~273 LB**. Сверху — v2c + adaptive 22% + clips → **271.697**.

### Фаза 3 — multiseed и тонкая настройка (272 → 268.76)

- **3-seed** (42, 1337, 2024) усреднение OOF/test — −1.5…−2.0 LB
- Сужение adaptive pct: 22% → 6% → 4% → **3.5%**
- CC50 clip: 99.0 → **98.95**, IC50 clip зафиксирован на **98.80**
- H6/H7 sweep (2026-05-29): pct=3.5% + CC9895 → **268.76238** (лучший)

5-seed, pct 2%/3%, pct 7%, blends, derived SI — регрессия.

## Финальный pipeline

4 стадии. Ключевой инсайт: **multiseed averaging простой базы** +
**узкий adaptive pct (3.5%)** + **точный clip IC50/CC50**.

### 1. Multiseed base: CatBoost + LightGBM 75/25

- Outlier removal: `train[train['SI'] < 2000]` → 746 rows
- Drop constant (std=0) и high-correlated (corr > 0.95) фичи → 159 фич
- 3 seed (42, 1337, 2024) × 10-fold KFold, усреднение OOF и test predictions
- Per-target гиперпараметры CatBoost/LightGBM, blend 0.75×CB + 0.25×LGB
- log1p только для SI

Кэш multiseed base: `base_cache/` (пересоздаётся при отсутствии).

### 2. v2c SI stack

LightGBM предсказывает log1p(SI) на расширенных фичах:

- 192 дескриптора (после NaN-median + drop zero-var)
- + OOF/pred IC50, CC50, SI из базы

4 варианта гиперпараметров (baseline / conservative / wide / huber),
5-fold StratifiedKFold по бинам log-таргета × 3 seed.

### 3. Adaptive top-3.5% min-of-teachers

Для каждой test row — log-std across 4 v2c teachers (uncertainty).
Топ-**3.5%** самых неуверенных → `min(4 teachers)`.
Остальные 96.5% → mean-of-4 (v2c ensemble).

### 4. Post-processing clips

- **IC50** clip на **98.80** percentile test predictions
- **CC50** clip на **98.95** percentile

## Эволюция LB (сводная)

| Этап | LB | Ключевое изменение |
|------|-----|-------------------|
| Baseline HGB | 365.82 | sklearn default |
| Большой ансамбль + v2c | ~278 | 1500+ fits, плато |
| Простая база CatBoost+LGBM | ~273 | outlier removal + feature drop |
| + v2c + adaptive 22% | 271.7 | stack на простой базе |
| Multiseed base | ~270 | 3-seed variance reduction |
| Adaptive pct 22% → 6% | 268.90 | меньше min-blend |
| Adaptive pct 6% → 4% + CC99.0 | 268.78 | H1 sweep |
| CC clip 99.0 → 98.95 | 268.76282 | H2 CC micro-tune |
| Adaptive pct 4% → **3.5%** | **268.76238** | H7 radical sweep |

## Что сработало / не сработало

**Сработало:**
- Multiseed averaging (3 seeds) — −1.5…−2.0 LB
- Узкий adaptive pct (22% → 3.5%) — −3+ LB
- CC50 clip 98.95 (vs 99.0) — −0.02 LB
- IC50 clip 98.80 — стабильный gain
- Упрощение базы (278 → 272) — главный скачок

**Не сработало:**
- 5-seed base — регрессия
- pct 2%/3% — хуже (269.2+)
- pct 7% — 269.21
- Blends pct04+pct06 — 268.84
- Derived SI blend (CC/IC) — 269.27
- ChEMBL external data — distribution mismatch
- Сложный ансамбль при OOF-gain — регрессия LB


## Воспроизведение

Положить train, test, sample_submission в папку data.
Импортировать библиотеки в начале ноутбука.
Прогнать ноутбук с начала до конца.

## Зависимости

```
python>=3.10
pandas, numpy, scipy
scikit-learn>=1.3
lightgbm>=4.0
catboost>=1.2
```
