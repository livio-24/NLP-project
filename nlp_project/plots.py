from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


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