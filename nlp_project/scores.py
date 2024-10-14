import pandas as pd
import json
import typer
import sys

from nlp_project.config import PROCESSED_DATA_DIR

app = typer.Typer()


def dataframe_scores(dataset, features_scores):
    "build dataframe containing all the scores for each product: star_rating, price, RSS, FSS, global_score"

    scores_df = dataset.groupby('asin')['BERT_RSS'].mean().reset_index(name='RSS')
    scores_df['star_rating'] = dataset.groupby('asin')['overall'].mean().reset_index()['overall']
    scores_df['price'] = dataset.groupby('asin')['price'].mean().reset_index()['price']
    scores_df['FSS'] = extract_FSS(features_scores)
    scores_df['global_score'] = calculate_global_score(scores_df)
    
    scores_df = scores_df.merge(
        dataset[['asin', 'title']].drop_duplicates(),
        on='asin',
        how='left'
    )

    scores_df.reset_index(drop=True, inplace=True)
    scores_df.to_csv(PROCESSED_DATA_DIR / 'scores_df.csv', index=False, sep=';')
    
    return scores_df


def extract_FSS(data):
    "extract the feature-based score (FSS) for each product from the json file"
    FSS = []
    for e in data:
        values = [f for f in list(e['features'].values()) ]
        mean = sum(values)/len(values) if len(values) != 0 else 1
        FSS.append(mean)

    return FSS

def calculate_global_score(scores_df):
    "apply formula to calculate the global_score, combining all the scores together"

    min_RSS = min(scores_df['RSS'])
    min_FSS = min(scores_df['FSS'])
    max_RSS = max(scores_df['RSS'])
    max_FSS = max(scores_df['FSS'])
    min_star = min(scores_df['star_rating'])
    max_star = max(scores_df['star_rating'])
    max_price = max(scores_df['price'])

    norm_RSS = (scores_df['RSS'] - min_RSS)/(max_RSS - min_RSS)
    norm_FSS = (scores_df['FSS'] - min_FSS)/(max_FSS - min_FSS)
    norm_star = (scores_df['star_rating'] - min_star)/(max_star - min_star)
    norm_price = (max_price - scores_df['price'] + 1)/max_price

    global_score = norm_price * 0.3 + norm_star * 0.2 + norm_RSS * 0.25 + norm_FSS * 0.25

    return global_score

@app.command()
def top_k_products_by_feature(feature: str , k: int):
    with open(PROCESSED_DATA_DIR / 'bert_features_scores.json', 'r') as file:
        features_scores = json.load(file)

    p = [(product['asin'],product['features'][feature]) for product in features_scores if feature in product['features'].keys()]
    top_K = sorted(p, key=lambda x: x[1], reverse=True)[:k]
    # Conversione della lista di coppie in DataFrame
    df = pd.DataFrame(top_K, columns=['asin', f'{feature}_score'])
    print(df)

@app.command()
def top_k_features(product_asin: str, k: int): 
    with open(PROCESSED_DATA_DIR / 'bert_features_scores.json', 'r') as file:
        features_scores = json.load(file)

    features = [product['features'] for product in features_scores if product['asin']==product_asin][0]
    sorted_features = {k:v for k, v in sorted(features.items(), key=lambda item: item[1], reverse=True)[:k]}
    # Convertire in DataFrame
    df = pd.DataFrame(sorted_features.items(), columns=['Feature', 'Score'])

    print(df)

if __name__ == '__main__':
    app()