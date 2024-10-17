# nlp-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Progetto per il corso di Natural Language Processing, Laurea Magistrale in Informatica, Università degli Studi di Salerno. 

Il progetto consiste nell'implementazione dell'architettura proposta nel seguente [paper](https://www.mdpi.com/2071-1050/11/15/4235), andando ad approfondire lo step di valutazione sperimentale.

## Installazione

1. Clonare il repository:
   ```bash
   https://github.com/livio-24/NLP-project.git

2. Assicurati di avere Conda installato. Puoi ricreare l'ambiente con il seguente comando:
    ```bash
    conda env create -f environment.yml
3. Attiva l'ambiente
   ```bash
    conda activate nlp-project
4. Scaricare i dati dal seguente [link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) (reviews e metadata), e inserirli nella cartella data/raw 

## Esempi d'uso
Una volta eseguiti gli step di installazione, è possibile lanciare il seguente script:
```bash
    python run_analysis.py
```
Lanciando questo script vengono eseguiti i vari step di analisi e si ottengono vari file in output tra cui:
- Il dataset preprocessato ottenuto dalla fase di preprocessing, viene salvato in data/processed/preprocessed_data.csv
- Il dataset finale con la colonna BERT_RSS, ossia il sentiment score per ogni review ottenuto applicando un modello bert, salvato in data/processed/final_scored_dataset.csv
- Il file json in cui vengono salvati per ogni prodotto i sentiment scores per ogni sentence, ed i sentiment scores per ciascuna feature, salvato in data/processed/bert_features_scores.json
- Il dataframe contenente i vari scores per ciascun prodotto (RSS,FSS,global score, price), salvato in data/processed/scores_df.csv

Una volta lanciato lo script ed ottenuto i vari file di output, è possibile lanciare vari scripts per effettuare delle query sui dati, per ora sono presenti i seguenti due comandi:
- Query per estrarre i top k prodotti in base ad una certa feature
  ```bash
    python scores.py top_k_products_by_feature feature k
    ```
- Query per estrarre le top k features di un dato prodotto
  ```bash
    python scrores.py top_k_features_by_product feature asin
    ```

## Valutazione sperimentale
E' stata condotta un’analisi su due variabili specifiche, la variabile temporale (anno recensione) e la variabile brand.

L’obiettivo di questa analisi è capire quali sono le features più citate nelle recensioni per ciascun dei 5 brand principali (Apple, Samsung, Blackberry, Motorola, LG) dal 2013 al 2018, e analizzare i trend di due features specifiche, batteria e schermo, anno per anno.

I risultati ottenuti sono salvati nella cartella reports/figures. E' possibile eseguire l'analisi eseguendo il notebook 05-lv-brand-analysis.

## Tecnologie utilizzate
- Nltk
- HuggingFace
- Plotly
- Matplotlib
- Typer
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nlp_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nlp_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nlp_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

