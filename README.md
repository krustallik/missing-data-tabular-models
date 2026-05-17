# INS: Tabular Models with Missing Data

Tento projekt skúma, ako rôzne ML modely pracujú s neúplnými tabuľkovými dátami v úlohe binárnej klasifikácie na bankruptcy datasetoch.

Hlavná myšlienka je porovnať klasické ML modely a tabular foundation models pri rôznych mechanizmoch chýbajúcich hodnôt, rôznych úrovniach missing data a rôznych stratégiách imputácie.

## Čo Robí Projekt

Projekt:

- používa 3 bankruptcy datasety: `polish_1year`, `slovak_manufacture_13`, `taiwan_bankruptcy`;
- vytvára train/test splits;
- umelo pridáva missing values podľa mechanizmov `MCAR`, `MAR`, `MNAR`;
- testuje rôzne stratégie spracovania chýbajúcich hodnôt;
- porovnáva classical models a foundation models;
- počíta metriky vhodné pre silný class imbalance;
- ukladá výsledky do CSV, markdown reports, logs a visualizations;
- obsahuje samostatný pipeline na hľadanie hyperparametrov klasických modelov cez randomized search.

Hlavná metrika pri tuningu parametrov je `PR-AUC` / `average_precision`, nie `accuracy`, pretože pozitívnej triedy je v datasetoch veľmi málo.

## Hlavné Pipeliny

### 1. Hlavný Experimentálny Pipeline

Súbor:

```text
project/src/run_experiments.py
```

Tento pipeline spúšťa celý experiment:

1. pripraví train/test splits;
2. overí missingness injection;
3. otestuje imputation methods;
4. spustí všetky modely na všetkých scenároch;
5. skonsoliduje výsledky;
6. vytvorí tabuľky, grafy a markdown reporty.

Modely:

- Classical: `Logistic Regression`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`, `SVM`, `MLP`, `CatBoost`;
- Foundation: `TabPFN`, `TabICL`.

Imputation methods:

- `mean`;
- `median`;
- `knn`;
- `mice`;
- `mice_indicator`;
- `none`.

### 2. Randomized Search Pipeline

Súbor:

```text
project/src/random_search_pipeline.py
```

Tento pipeline hľadá dobré hyperparametre pre 8 klasických modelov:

- `logistic_regression`;
- `random_forest`;
- `gradient_boosting`;
- `xgboost`;
- `lightgbm`;
- `svm`;
- `mlp`;
- `catboost`.

Pipeline:

- generuje `150` náhodných kombinácií parametrov pre každý model;
- používa 3 random seeds;
- používa `StratifiedKFold(n_splits=5)`;
- optimalizuje `average_precision` / `PR-AUC`;
- dodatočne zapisuje `balanced_accuracy`;
- nájde najlepšie parametre samostatne pre každý dataset;
- nájde univerzálne parametre, ktoré v priemere fungujú najlepšie na všetkých troch datasetoch;
- podporuje `--resume`, aby bolo možné pokračovať po zastavení.

## Ako Spustiť Projekt Po Stiahnutí Z Git

### 1. Prejsť Do Priečinka Projektu

```powershell
cd D:\INS\project
```

Ak je repozitár stiahnutý do iného priečinka, treba prejsť do vlastného priečinka `project`.

### 2. Vytvoriť Virtual Environment

```powershell
python -m venv .venv
```

Aktivovať ho:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Aktualizovať Pip

```powershell
python -m pip install --upgrade pip
```

### 4. Nainštalovať Závislosti

```powershell
pip install -r requirements.txt
```

### 5. Skontrolovať, Či Sú Dáta Na Mieste

Očakávané processed datasety:

```text
project/data/processed/polish_1year.csv
project/data/processed/slovak_manufacture_13.csv
project/data/processed/taiwan_bankruptcy.csv
```

Očakávané train/test splits:

```text
project/data/splits/
```

Ak splits chýbajú, hlavný pipeline ich vytvorí automaticky.

## Ako Spúšťať Hlavný Pipeline

Plné spustenie:

```powershell
python src\run_experiments.py
```

Spustenie iba hlavnej experiment loop časti:

```powershell
python src\run_experiments.py --step 4
```

Spustenie od konkrétneho kroku až do konca:

```powershell
python src\run_experiments.py --from 4
```

Príklad spustenia iba pre jeden dataset:

```powershell
python src\run_experiments.py --step 4 --datasets taiwan_bankruptcy
```

Príklad spustenia iba pre niekoľko modelov:

```powershell
python src\run_experiments.py --step 4 --models xgboost lightgbm catboost
```

### Resume / pokračovanie hlavného pipelinu

Step 4 (`run_experiments`) si pamätá, čo už natrénoval. Pri každej dokončenej kombinácii zapisuje nový riadok do hlavného CSV:

```text
project/results/tables/experiment_results.csv
```

Zápis prebieha atomicky cez `experiment_results.csv.tmp` → `Path.replace(...)`, takže prerušenie behu nikdy nezanechá poškodený CSV.

Jedna „kombinácia" je identifikovaná 7-ticou:

```text
(dataset, split_seed, seed, missing_mechanism, missing_rate, imputation, model)
```

Pre native scenár sa do CSV zapisuje stabilná hodnota `missing_mechanism = "native"` a `missing_rate = 0.0` (nikdy `NaN`), aby porovnanie pri resume bolo deterministické. Staršie CSV s `NaN` pre native sa pri načítaní automaticky znormalizujú.

Pokračovať po prerušení (default):

```powershell
python src\run_experiments.py --step 4 --resume
```

`--resume` je zapnutý štandardne, takže rovnakú vec robí aj:

```powershell
python src\run_experiments.py
python src\run_experiments.py --step 4
python src\run_experiments.py --from 4
```

Ignorovať existujúci CSV a spustiť všetko nanovo:

```powershell
python src\run_experiments.py --step 4 --no-resume
```

Pretrénovať aj kombinácie, ktoré sú už v CSV (bez vymazania súboru):

```powershell
python src\run_experiments.py --step 4 --force
```

Pri zapnutom resume pipeline na začiatku kroku 4 zaloguje:

```text
Resume: loaded N rows from experiment_results.csv (K unique completed keys)
Already done : K keys
```

a počas behu pre každú už hotovú kombináciu vypíše `SKIP completed` (debug) alebo zhrnuté `scenario fully cached ... skipping` (info). Nové tréningy logujú `RUN ...` a `SAVED ...` po každom modeli.

## Ako Spúšťať Randomized Search

Plný deep randomized search:

```powershell
python src\random_search_pipeline.py --n-iter 150 --resume
```

Význam flagov:

- `--n-iter 150` - 150 náhodných kombinácií parametrov pre každý model;
- `--resume` - ak sa beh zastaví, ďalšie spustenie preskočí už dokončené dvojice `(dataset, model)`.

Rýchly test, že pipeline funguje:

```powershell
python src\random_search_pipeline.py --models logistic_regression --datasets slovak_manufacture_13 --seeds 42 --n-iter 5
```

Spustenie iba rýchlejších modelov:

```powershell
python src\random_search_pipeline.py --n-iter 150 --resume --models logistic_regression random_forest gradient_boosting xgboost lightgbm catboost
```

Spustenie SVM samostatne:

```powershell
python src\random_search_pipeline.py --n-iter 50 --resume --models svm
```

## Aké Výsledky Budú Na Konci

### Výsledky Hlavného Pipelinu

Hlavné CSV:

```text
project/results/tables/experiment_results.csv
project/results/tables/consolidated_results.csv
project/results/tables/classical_models.csv
project/results/tables/foundation_models.csv
project/results/tables/robustness_analysis.csv
```

Reports:

```text
project/results/reports/
```

Visualizations:

```text
project/results/visualizations/
```

Logs:

```text
project/results/logs/
```

### Výsledky Randomized Search

Všetky výsledky random search:

```text
project/results/tables/random_search/random_search_full_results.csv
```

Najlepšie parametre pre každý model samostatne na každom datasete:

```text
project/results/tables/random_search/random_search_top_per_dataset.csv
```

Najuniverzálnejšie parametre pre každý model naprieč všetkými tromi datasetmi:

```text
project/results/tables/random_search/random_search_universal.csv
```

Logs random search:

```text
project/results/logs/
```

## Odporúčaný Postup Práce

1. Nainštalovať závislosti:

```powershell
cd D:\INS\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Spustiť random search na hľadanie dobrých parametrov:

```powershell
python src\random_search_pipeline.py --n-iter 150 --resume
```

3. Pozrieť si nájdené parametre:

```text
project/results/tables/random_search/random_search_top_per_dataset.csv
project/results/tables/random_search/random_search_universal.csv
```

4. Spustiť hlavný experimentálny pipeline:

```powershell
python src\run_experiments.py
```

5. Analyzovať finálne výsledky:

```text
project/results/tables/
project/results/reports/
project/results/visualizations/
```

