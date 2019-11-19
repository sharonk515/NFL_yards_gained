import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def two_subplots(df, col1, col2, size, num_type=1, order=None, rotate=None):
    fig, axes = plt.subplots(1, 2, figsize=size)
    plt.subplots_adjust(wspace=0.4)
    sns.countplot(df[col2], order=order, ax=axes[1])
    if num_type == 2:
        sns.distplot(df[col1], ax=axes[0])
    else:
        sns.countplot(df[col1], order=order, ax=axes[0])
    axes[0].set_title(f'Distribution of {col1}')
    axes[1].set_title(f'Distribution of {col2}')
    if rotate:
        plt.setp(axes[1].get_xticklabels(), rotation=90);

def plot_distributions(df, col_list):
    fig, axes = plt.subplots(6, 3, figsize=(16, 20))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    for i, feat in enumerate(col_list):
        sns.distplot(df[feat], ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Distribution of {feat}');


def plot_boxplots(df, col_list):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, col in enumerate(col_list):
        sns.boxplot(df[col], df['Yards'], ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Distribution of Yards for each {col}');
