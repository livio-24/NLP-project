from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import gzip
import json
import pandas as pd
from afinn import Afinn
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords

from nlp_project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Inizializza le risorse una sola volta
afinn = Afinn()
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('universal_tagset')


def parse(path):
    """Parse a gzipped JSON file line by line."""
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def getDF(path):
    """Load a JSON file into a pandas DataFrame."""
    df = {}
    for i, d in enumerate(parse(path)):
        df[i] = d
    return pd.DataFrame.from_dict(df, orient='index')

def load_data():
    """Load datasets from specified paths."""
    path = RAW_DATA_DIR / 'Cell_Phones_and_Accessories.json.gz'
    path_meta = RAW_DATA_DIR / 'meta_Cell_Phones_and_Accessories.json.gz'
    
    data = getDF(path)
    meta = getDF(path_meta)
    
    return data, meta

def preprocess(df, column):
    afinn = Afinn() 
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    preprocessed_text = []
    tagged_text = []
    for row in tqdm(df[column], total = len(df[column])):
        afinn_scores = []
        text_cleaned = re.sub(r'[^\w\s]', '', row) #cleaning: remove all character that aren't whitespaces or alphanumeric
        words = nltk.word_tokenize(text_cleaned) #tokenization
        words = [w.lower() for w in words] #lower casing
        tagged_words = nltk.pos_tag(words, tagset='universal') #POS-tagging
        preprocessed_words = [(lemmatizer.lemmatize(w), tag) for w, tag in tagged_words if not w in stop_words] #stopwords removal and lemmatization
        preprocessed_text.append(' '.join([w for w, tag in preprocessed_words]))
        #afinn scores
        for word, tag in preprocessed_words:
            score = afinn.score(word)
            afinn_scores.append((word, tag, score))
        tagged_text.append(afinn_scores)

    return preprocessed_text, tagged_text

def build_ontolgy(df):
    nouns = []
    for index, row in tqdm(df.iterrows(), total = len(df)):
        text = row['description']
        text_cleaned = re.sub(r'[^\w\s]', '', text) #cleaning
        words = nltk.word_tokenize(text_cleaned) #tokenization
        words = [w.lower() for w in words] #lower casing
        tagged_words = nltk.pos_tag(words, tagset='universal') #POS-tagging
        preprocessed_words = [(lemmatizer.lemmatize(w), tag) for w, tag in tagged_words if not w in stop_words] #stopwords removal and lemmatization
        #print(filter_text)
        for word, tag in preprocessed_words:
            if(tag == 'NOUN'):
                nouns.append(word)
            
    return nouns

