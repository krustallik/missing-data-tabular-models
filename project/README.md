# Semestrálny projekt: Hodnotenie TabPFN a TabICL pri chýbajúcich dátach

Tento repozitár obsahuje semestrálny projekt zameraný na porovnanie pretrained tabular foundation modelov **TabPFN** a **TabICL** s klasickými modelmi pri práci s neúplnými dátami v klasifikačných úlohách.

## Čo bolo cieľom

- vyhodnotiť správanie modelov pri mechanizmoch chýbania **MCAR, MAR, MNAR**,
- porovnať viacero prístupov k chýbajúcim hodnotám (mean/median, kNN, MICE, indikátory chýbania, implicitné spracovanie),
- posúdiť robustnosť modelov pri rastúcej miere chýbajúcich hodnôt,
- formulovať praktické odporúčania pre použitie v reálnych úlohách.

## Použité datasety

- Polish Companies (1-year),
- Slovak Manufacture 13,
- Taiwan Bankruptcy.

## Hlavný výstup pre hodnotenie

Finálna technická správa je v koreňovom adresári projektu:

- `../Hodnotenie_TabFM_pri_chybajucich_datach.docx`
- (alternatívne novšia verzia pri otvorenom súbore) `../Hodnotenie_TabFM_pri_chybajucich_datach_v2.docx`

## Kde sú podporné výsledky

- `results_*/tables/` - agregované tabuľky výsledkov,
- `results_*/visualizations/` - grafy použité v správe,
- `results_*/reports/` - automaticky generované textové reporty.

## Reprodukovateľnosť (stručne)

Po inštalácii závislostí je možné experimenty spustiť cez skripty v `src/`.
Projekt je pripravený tak, aby boli výsledky reprodukovateľné (fixný `random_state` a jednotný experimentálny protokol).

