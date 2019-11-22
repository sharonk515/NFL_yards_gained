import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix


def two_subplots(df, col1, col2, size, num_type=1, order=None, rotate=None):
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
    if model == 'SVM':
        plt.barh(df['Feature'],
                 df['Absolute Coefficient'],
                 align='center')
        sns.despine(left=False, bottom=False)
        coefs = df['Coefficient'].apply(lambda x: round(x, 2))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=18)
        plt.xlabel('Feature importance', fontsize=18)
        plt.ylabel('Feature', fontsize=18)
        plt.title(f'Predicting Yards\n', fontsize=18)
        for i, v in enumerate(coefs):
            plt.text(np.abs(v) + 0.007, i - 0.1,
                     str(np.abs(v)),
                     color='black',
                     fontsize=18)
            if v > 0:
                plt.text(v - 0.01, i - 0.1, '+',
                         color='black',
                         fontsize=12,
                         fontweight='bold')
            else:
                plt.text(np.abs(v) - 0.03, i - 0.1, '-',
                         color='black',
                         fontsize=18,
                         fontweight='bold')
    else:
        n_features = df.shape[1]
        plt.barh(range(n_features),
                 model.best_estimator_.feature_importances_,
                 align='center')
        plt.yticks(np.arange(n_features), df.columns.values)
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.title(f'Feature importance in predicting yards');


def plot_roc_curve(models, model_names, X_test, y_test, X_test_scale=None):
    '''
    This function plots ROC curves for each model specified.

    Parameters
    ----------
    models: list of models
    model_name: list, model names as strings
    X_test: DataFrame or series, test set of features
    y_test: DataFrame or series, test set of target values
    X_test_scale: df (default=None), scaled DataFrame
                  for models that require scaling

    Returns
    -------
    ROC curve of all of the models with their areas under the curve
    '''
    plt.figure(figsize=(10, 8))
    for idx, model in enumerate(models):
        if (model_names[idx] == 'SVM'):
            auc_score = roc_auc_score(
                y_test, model.best_estimator_.predict_proba(X_test_scale)[:, 1])
            fpr, tpr, thresholds = roc_curve(
                y_test, model.best_estimator_.predict_proba(X_test_scale)[:, 1])
        else:
            auc_score = roc_auc_score(
                y_test, model.best_estimator_.predict_proba(X_test)[:, 1])
            fpr, tpr, thresholds = roc_curve(
                y_test, model.best_estimator_.predict_proba(X_test)[:, 1])
        # fpr = false positive, #tpr = true positive
        plt.plot(fpr, tpr,
                 label=f'{model_names[idx]} (auc = %0.2f)' % auc_score,
                 lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 0, 1], [0, 1, 1], 'k--', color='red')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'ROC Curves', fontsize=20)
    plt.xlim(-0.005, 1.005)
    plt.ylim(-0.005, 1.005)
    plt.legend(loc='best', fontsize=12, frameon=False);
