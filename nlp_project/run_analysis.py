from nlp_project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from nlp_project.dataset import *
from nlp_project.sentiment_analysis import *

from afinn import Afinn
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Funzione per rilevare la lingua
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    
def main():
    #STEP 1
    dataset = generate_dataset()
    #print(dataset.head())

    #STEP 2
    dataset.dropna(subset=['price', 'reviewText'], inplace=True)
    # regex per il formato corretto dei prezzi
    regex = re.compile(r'^\$\d+\.\d+$')
    # Filtriamo i dati applicando la regex alla colonna 'price'
    dataset = dataset[dataset['price'].apply(lambda x: bool(regex.match(x)))]
    # Rimuoviamo il simbolo del dollaro dalla colonna 'prezzi'
    dataset['price'] = dataset['price'].str.replace('$', '', regex=False).astype(float)
    reviews_count_per_product = dataset.groupby('asin').size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    less_100_reviews = reviews_count_per_product[reviews_count_per_product['counts'] < 100]['asin']
    dataset = dataset[~dataset['asin'].isin(less_100_reviews)]
    # Per risultati più consistenti con langdetect
    DetectorFactory.seed = 0
    # Rileva la lingua di ogni recensione
    dataset['language'] = dataset['reviewText'].apply(detect_language)
    dataset = dataset[~dataset['language'].isin(['es', 'pt'])]
    dataset['preprocessed_text'], dataset['tagged_text'] = preprocess(dataset, 'reviewText')
    dataset.drop(columns=['feature', 'language'], inplace=True)

    dataset.to_csv(PROCESSED_DATA_DIR / 'preprocessed_data.csv', index=False, sep=';')

    #STEP 3
    ontology_filtered = pd.read_csv(PROCESSED_DATA_DIR / 'ontology_filtered.csv')

    #STEP 4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)
    dataset['BERT_RSS'] = dataset['reviewText'].progress_apply(sentiment_score_bert, args=(tokenizer, model))
    dataset.to_csv(PROCESSED_DATA_DIR / 'final_scored_dataset.csv', index=False, sep=';')

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #sentence_tokenizer
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    afinn = Afinn()

    json_sentences_and_features_scores(dataset,
                                       ontology_filtered,
                                       lemmatizer,
                                       stop_words,
                                       afinn,
                                       sent_tokenizer,
                                       save_path = f'bert_features_scores.json',
                                       sa='bert')
    

if __name__=='__main__':
    main()