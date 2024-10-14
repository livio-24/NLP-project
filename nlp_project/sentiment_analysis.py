import pandas as pd
#import torch
import re 
import nltk
import json

from nlp_project.config import PROCESSED_DATA_DIR
from nlp_project.dataset import cleaning_steps

#----------------------------- BERT ---------------------------------------------------------------

def get_input_ids_and_attention_mask_chunk(tokens, device):
    """
    This function splits the input_ids and attention_mask into chunks of size 'chunksize'. 
    It also adds special tokens (101 for [CLS] and 102 for [SEP]) at the start and end of each chunk.
    If the length of a chunk is less than 'chunksize', it pads the chunk with zeros at the end.
    
    Returns:
        input_id_chunks (List[torch.Tensor]): List of chunked input_ids.
        attention_mask_chunks (List[torch.Tensor]): List of chunked attention_masks.
    """
    chunksize = 512
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    attention_mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))
    
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([
            torch.tensor([101], device=device), input_id_chunks[i], torch.tensor([102], device=device)
        ])
        
        attention_mask_chunks[i] = torch.cat([
            torch.tensor([1], device=device), attention_mask_chunks[i], torch.tensor([1],device=device)
        ])
        
        pad_length = chunksize - input_id_chunks[i].shape[0]
        
        if pad_length > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.tensor([0] * pad_length, device=device)
            ])
            attention_mask_chunks[i] = torch.cat([
                attention_mask_chunks[i], torch.tensor([0] * pad_length, device=device)
            ])
            
    return input_id_chunks, attention_mask_chunks 

def sentiment_score_bert(review, tokenizer, model, device):
    tokens = tokenizer.encode_plus(review, add_special_tokens=False, return_tensors='pt').to(device)
    input_id_chunks, attention_mask_chunks = get_input_ids_and_attention_mask_chunk(tokens)
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(attention_mask_chunks)
    input_dict = {
        'input_ids' : input_ids.long(),
        'attention_mask' : attention_mask.int()
    }
    
    outputs = model(**input_dict)
    probabilities = torch.nn.functional.softmax(outputs[0], dim = -1)
    mean_probabilities = probabilities.mean(dim = 0)
    
    return torch.argmax(mean_probabilities).item() + 1

#----------------------- SENTENCE LEVEL ANALYSIS ---------------------------------------

# Funzione per splittare le recensioni in frasi
def preprocess_analyze_sentences(review, lemmatizer, sent_tokenizer, stop_words, afinn, sa):
    sentences_sentiments = {}
    sentences = sent_tokenizer.tokenize(review)
    
    for sentence in sentences:
        afinn_scores = []
        preprocessed_words = cleaning_steps(text=sentence, lemmatizer=lemmatizer, stop_words=stop_words)
        #sentence_clean = re.sub(r'[^\w\s]', '', sentence) #cleaning
        # Tokenizza in parole
        #words = nltk.word_tokenize(sentence_clean)
        # Rimuovi stopwords e applica lemmatization
        #preprocessed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in stop_words]

        for word, _ in preprocessed_words:
            score = afinn.score(word)
            if score != 0:
                afinn_scores.append(score)
        # Ricostruisci la frase preprocessata come stringa
        preprocessed_sentence = ' '.join([w for w, tag in preprocessed_words])
        # Calcola il sentiment
        if(sa == 'afinn'):
            sentences_sentiments[preprocessed_sentence] = normalize_RSS((sum(afinn_scores)/len(afinn_scores))) if len(afinn_scores) != 0 else normalize_RSS(0)
        elif(sa == 'bert'):
            sentences_sentiments[preprocessed_sentence] = sentiment_score_bert(sentence)
        
# Funzione per combinare tutti i dizionari di sentences per ogni 'asin'
def combine_sentences(sent_dicts):
    combined_dict = {}
    for d in sent_dicts:
        combined_dict.update(d)
    return combined_dict

# Aggiungere il campo features basato sulla lista di features
def extract_features(sent_dict, features_list):
    feature_sentiments = {}
    for feature in features_list:
        feature_values = [sentiment for sentence, sentiment in sent_dict.items() if feature in sentence.split()]
        if feature_values:
            feature_sentiments[feature] = sum(feature_values) / len(feature_values)
    return feature_sentiments

def json_sentences_and_features_scores(dataset, features, lemmatizer, stop_words, afinn, sent_tokenizer, save_path, sa='bert'):
    # Applicare la funzione a ogni recensione e raggruppare per 'asin'
    dataset['preprocessed_sentences'] = dataset['reviewText'].progress_apply(preprocess_analyze_sentences, args=(lemmatizer, sent_tokenizer, stop_words, afinn, sa,))

    # Raggruppare per 'asin' e combinare le sentences
    grouped = dataset.groupby('asin')['preprocessed_sentences'].progress_apply(list).reset_index()
    grouped['preprocessed_sentences'] = grouped['preprocessed_sentences'].progress_apply(combine_sentences)
    grouped['features'] = grouped['preprocessed_sentences'].progress_apply(lambda x: extract_features(x, features['value']))

    result = grouped.to_dict(orient='records')
    #Salva il risultato in un file JSON
    with open(PROCESSED_DATA_DIR / save_path, 'w') as f:
        json.dump(result, f, indent=4)

#------------------AFINN--------------------------------
def calculate_review_sentiment_score(x):
    tot = 0
    card = 0
    for word, _, score in x:
        if score != 0:
            tot += score
            card += 1
    
    return tot/card if card != 0 else 0    

def normalize_RSS(x):
    norm = 0
    if x >= 3:
        norm = 5

    elif 1 < x <= 3:
        norm = 4

    elif -0.5 < x <= 1:
        norm = 3

    elif -3 < x <= -0.5:
        norm = 2

    elif x <= -3 :
        norm = 1

    return norm

#--------------------------------------------

        