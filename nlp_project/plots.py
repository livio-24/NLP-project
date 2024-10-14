from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import json

from nlp_project.config import FIGURES_DIR, PROCESSED_DATA_DIR

def sb_bar_plot(x,y, title, xlabel, ylabel, orient='h', save_path=None):
    "bar plot using Seaborn"
    plt.figure(figsize=(8, 6))
    sns.barplot(x=x, y=y, orient=orient)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Se l'orientamento è verticale, ruota le etichette dell'asse x
    if orient == 'v':
        plt.xticks(rotation=90)  # Ruota le etichette dell'asse x di 90 gradi
     # Se è stato specificato un percorso, salva il grafico
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plotly_ex_barplot(data,x,y,color,height=400):
    "barplot using plotly express"
    fig = px.bar(data_frame=data, x=x, y=y, color=color, height=height)
    fig.show()

def pie_chart_go(labels, values): 
    "Use `hole` to create a donut-like pie chart in plotly"
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.show()

def double_trace_barplot(x_trace1, y_trace1, x_trace2, y_trace2, text_trace1, text_trace2 , name_trace1, name_trace2, title):
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x_trace1, y=y_trace1,
                    #base=[-500,-600,-700],
                    #marker_color='green',
                    text=text_trace1,
                    name=name_trace1,
                    #texttemplate='%{text:.2f}',
                    textposition='auto',
                    ))
    fig.add_trace(go.Bar(x=x_trace2, y=y_trace2,
                    #base=[-500,-600,-700],
                    #marker_color='green',
                    text=text_trace2,
                    name=name_trace2,
                    #texttemplate='%{text:.2f}',
                    textposition='auto',
                    ))


    fig.update_layout(
        title = title
    )
    fig.show()

def create_circular_mask(diameter):
    x, y = np.ogrid[:diameter, :diameter]
    center = (diameter - 1) / 2
    mask = (x - center) ** 2 + (y - center) ** 2 > center ** 2
    return 255 * mask.astype(int)


def plot_features_wc(asin):

    with open(PROCESSED_DATA_DIR / 'bert_features_scores.json', 'r') as file:
        features_scores = json.load(file)

    circular_mask = create_circular_mask(diameter=400)
    
    for product in features_scores:
        if(product['asin'] == asin):
            pos_features = {k: v for k, v in product['features'].items() if v > 3}
            neg_features = {k: v for k, v in product['features'].items() if v < 3}

    # Definizione della palette di colori personalizzata
    colors = ["#FF5733", "#33FF57", "#3357FF"]
    cmap = LinearSegmentedColormap.from_list("custom_palette", colors)
    
    # Generate the word cloud
    pos_wordcloud = WordCloud(mask=circular_mask, relative_scaling=0,width=600, height=400, colormap=cmap, background_color='white').generate_from_frequencies(frequencies=pos_features)
    neg_wordcloud = WordCloud(mask=circular_mask, relative_scaling=0,width=600, height=400,colormap=cmap, background_color='white').generate_from_frequencies(frequencies=neg_features)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(pos_wordcloud, interpolation='bilinear')
    axs[0].axis('off')  # Nascondi gli assi
    axs[0].set_title('positive features')

    axs[1].imshow(neg_wordcloud, interpolation='bilinear')
    axs[1].axis('off')  # Nascondi gli assi
    axs[1].set_title('negative features')
    
    plt.tight_layout()
    plt.show()