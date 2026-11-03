# Dataset attribution

The processed CSVs in `project/data/processed/` are derived from three public bankruptcy datasets. The upstream dataset records specify the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Credit the original providers when sharing or adapting these data. The code in this repository has no selected license; the data licenses do not license the project code.

## Polish companies bankruptcy

The repository uses the one-year case (`1year.arff`) from the [UCI Polish Companies Bankruptcy dataset](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata).

> Tomczak, S. (2016). *Polish Companies Bankruptcy* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600.

## Slovak manufacturing bankruptcy

The repository uses the 2013 manufacturing bankrupt and non-bankrupt files from [Dataset of financial ratios of Slovak companies, Mendeley Data, version 2](https://data.mendeley.com/datasets/j89csb932y/2). The local filenames match those listed in the source release. The authors describe 21 financial ratios recorded for each of three years before evaluation. The prepared CSV contains 63 source predictors plus an unnamed index artifact; the experiment data loader removes that artifact before model fitting.

> Drotár, P., Gnip, P., Zoričak, M., & Gazda, V. (2019). *Small- and medium-enterprises bankruptcy dataset*. Data in Brief, 25, 104360. https://doi.org/10.1016/j.dib.2019.104360.
>
> Drotár, P., Gnip, P., Zoričak, M., & Gazda, V. (2019). *Dataset of financial ratios of Slovak companies* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/j89csb932y.2.

The Mendeley record identifies CC BY 4.0. The related data article is open access under CC BY 4.0 as well. The original records were compiled from Slovak SME annual reports, as described in the article. The repository's project-specific target mapping is implemented in `project/src/prepare_raw_datasets.py`: class 1 is assigned to the file named `bankrupt_manufacture_13_year_10_11_12.csv`, and class 0 to its `nonbankrupt` counterpart.

## Taiwan bankruptcy prediction

The repository uses the [UCI Taiwanese Bankruptcy Prediction dataset](https://archive.ics.uci.edu/dataset/572/taiwanese%2Bbankruptcy%2Bprediction).

> *Taiwanese Bankruptcy Prediction* [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5004D.

## Scope of attribution

These citations cover dataset provenance and license information. They do not determine a license for the project's Python code, documentation, generated figures, or other repository content. A code license should be selected by the project authors separately.
