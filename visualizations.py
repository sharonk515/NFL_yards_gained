import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def two_subplots(df, col1, col2, size, num_type=1,
                 order=None, rotate=None):
    '''
    This function plots two subplots.

    Parameters
    ----------
    df: DataFrame
    col1: name of column to use for first subplot
    col2: name of column to use for second subplot
    size: tuple, (width, length)
    num_type: int (1, 2, default=1) number of different types of subplots
    order: list (default=None) list of values to order by
    rotate: boolean (default=None)
            True if you want x-labels rotated by 90 degrees

    Returns
    -------
    Two countplots or 1 countplot and 1 distplot,
    displaying distributions of specified columns.
    '''
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
    '''
    This function plots distributions of all columns in the specified list.

    Parameters
    ----------
    df: DataFrame
    col_list: list, names of columns to plot

    Returns
    -------
    Distplots of all of the columns in the specified list
    '''
    fig, axes = plt.subplots(6, 3, figsize=(16, 20))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    for i, feat in enumerate(col_list):
        sns.distplot(df[feat], ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Distribution of {feat}');


def plot_boxplots(df, col_list):
    '''
    This function plots distributions of all columns in the specified list.

    Parameters
    ----------
    df: DataFrame
    col_list: list, names of columns to plot

    Returns
    -------
    Boxplots of all of the columns in the specified list
    '''
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, col in enumerate(col_list):
        sns.boxplot(df[col], df['Yards'], ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Distribution of Yards for each {col}');


def plot_group_count(df, column):
    yard_order = ['< 0', '0-1', '2-3', '4-6', '> 6']
    yard_counts = {}
    for i in yard_order:
        yard_counts[i] = dict(df[column].value_counts(sort=False))[i]
    sns.countplot(df[column],
                  order=yard_order)
    for i, v in enumerate(yard_counts.values()):
        plt.text(i - 0.27, v + 2000,
                 str(v),
                 color='black',
                 fontsize=10)
    plt.title('Distribution of Yard Groups \n')
    sns.despine(left=False, bottom=False)
    plt.savefig('Images/Yard_groups',
                bbox_inches='tight',
                transparent=True);


def plot_reg_feat_imp(model, df, top10=None):
    '''
    This function plots a horizontal bar graph,
    displaying the importance of features for specified model.

    Parameters
    ----------
    model: model to plot
    df: DataFrame
    top10: boolean (default=False), True if plotting Top 10 features

    Returns
    -------
    Horizontal bar graph showing the importance of features.
    '''

    feat_imp = pd.DataFrame([df.columns.values,
                             model.feature_importances_]
                           ).T
    feat_imp.columns = ['features', 'importance']
    feat_imp.sort_values(by='importance', inplace=True)
    if top10:
        plt.figure(figsize=(10, 7))
        plt.barh(feat_imp['features'][-10:],
                 feat_imp['importance'][-10:],
                 align='center')
        plt.yticks(np.arange(len(feat_imp['features'][-10:])),
                   feat_imp['features'][-10:])
        plt.title('Top 10 important features in predicting yards')

    else:
        plt.figure(figsize=(10, 10))
        plt.barh(feat_imp['features'],
                 feat_imp['importance'],
                 align='center')
        plt.yticks(np.arange(len(feat_imp['features'])),
                   feat_imp['features'])
        plt.title('Feature importance in predicting yards')

    plt.xlabel('Feature importance')
    plt.ylabel('Feature');


def plot_feat_imp(model, df):
    '''
    This function plots a horizontal bar graph,
    displaying the importance of features for specified model.

    Parameters
    ----------
    model: model to plot
    df: DataFrame

    Returns
    -------
    Horizontal bar graph showing the importance of features.
    '''
    plt.figure(figsize=(8, 8))
    if (model == svm) | (model == log):
        feat_imp = pd.DataFrame([df.columns.values,
                                 model.best_estimator_.coef_[0],
                                 np.abs(model.best_estimator_.coef_[0])],
                                columns=['features',
                                         'coefficients',
                                         'absolute_coefs']
                               )
        feat_imp.sort_values(by='absolute_coefs', inplace=True)
        plt.barh(feat_imp['features'],
                 feat_imp['absolute_coefs'],
                 align='center')
        plt.yticks(np.arange(len(feat_imp['features'])),
                   feat_imp['features'])
        sns.despine(left=False, bottom=False)
        coefs = df['coefficients'].apply(lambda x: round(x, 2))
        for i, coef in enumerate(coefs):
            plt.text(np.abs(coef) + 0.007, i - 0.1,
                     str(np.abs(coef)),
                     color='black',
                     fontsize=18)
            if v > 0:
                plt.text(coef - 0.01, i - 0.1, '+',
                         color='black',
                         fontsize=12,
                         fontweight='bold')
            else:
                plt.text(np.abs(coef) - 0.03, i - 0.1, '-',
                         color='black',
                         fontsize=18,
                         fontweight='bold')

    else:
        feat_imp = pd.DataFrame([df.columns.values,
                                 model.best_estimator_.feature_importances_],
                               ).T
        feat_imp.columns=['features', 'importance']
        feat_imp.sort_values(by='importance', inplace=True)
        plt.barh(feat_imp['features'],
                     feat_imp['importance'],
                     align='center')
        plt.yticks(np.arange(len(feat_imp['features'])),
                       feat_imp['features'])

    plt.title('Feature importance in predicting yards')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature');
