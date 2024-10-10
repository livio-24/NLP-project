from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from nlp_project.config import FIGURES_DIR, PROCESSED_DATA_DIR

def sb_bar_plot(x,y, title, xlabel, ylabel, orient='h', save_path=None):
    "bar plot using Seaborn"
    plt.figure(figsize=(8, 6))
    sns.barplot(x, y, orient)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
     # Se è stato specificato un percorso, salva il grafico
    if save_path:
        plt.savefig(save_path)
    plt.show()

