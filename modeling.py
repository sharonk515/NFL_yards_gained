import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.SVM import SVC
import catboost as cb
import pickle

def run_randomized_search(model, model_name, params, X_train, X_test,
                          y_train, y_test, cat_feats=None):

    '''
    This function runs Randomized Search with various models.

    Parameters
    ----------
    model: function for model
    (e.g. RandomForestClassifier, CatBoost)
    model_name: str, model name as a string
    params: dict, different parameters
    X_train: DataFrame, train set of variables
    X_test: DataFrame, test set of variables
    y_train: DataFrame, train set of target values
    y_test: DataFrame, test set of target values
    random_state: int

    Returns
    --------
    Model with the best hyperparameters, train score, and test score.
    '''

    if model == cb.CatBoostClassifier:
        rs = RandomizedSearchCV(estimator=model(loss_function='MultiClass',
                                                random_state=220),
                                param_distributions=params,
                                scoring='accuracy',
                                cv=5, verbose=1, n_jobs=-1)

        mod = rs.fit(X_train, y_train, cat_features=cat_feats)

    else:
        rs = RandomizedSearchCV(estimator=model(random_state=220),
                                param_distributions=params,
                                scoring='accuracy',
                                cv=5, verbose=1, n_jobs=-1)

        mod = rs.fit(X_train, y_train)

    pickle.dump(mod, open(f'Models/{model_name}.sav', 'wb'))

    print('Best params:', rs.best_params_)
    print('Train score: %.3f' % rs.best_score_)
    print('Test score: %.3f' % rs.score(X_test, y_test))

    return mod


def run_randomized_search_scaled(model, model_name, params,
                                 X_train_scale, X_test_scale,
                                 y_train, y_test, random_state=None):
    '''
    This function runs Randomized Search with standardized data.

    Parameters
    ----------
    model: function for model that requires scaling
    (e.g. LogisticRegression, KNeighborsClassifier, SVC)
    model_name: str, model name as a string
    params: dict, different parameters
    X_train_scale: DataFrame, scaled train set of variables
    X_test_scale: DataFrame, scaled test set of variables
    y_train: DataFrame, train set of target values
    y_test: DataFrame, test set of target values
    random_state: int

    Returns
    --------
    Model with the best hyperparameters, train score, and test score.
    '''
    # if model == LogisticRegression:
    rs = RandomizedSearchCV(estimator=model(solver='saga',
                                            multi_class='multinomial',
                                            random_state=random_state),
                            param_distributions=params,
                            scoring='accuracy',
                            cv=5, verbose=1, n_jobs=-1)

    # elif model == SVC:
    #     rs = RandomizedSearchCV(estimator=SVC(probability=True,
    #                                           decision_function_shape='ovo',
    #                                           random_state=random_state),
    #                             param_distributions=params,
    #                             scoring='accuracy',
    #                             cv=5, verbose=1, n_jobs=-1)
    #
    # else:
    #     rs = RandomizedSearchCV(estimator=model(),
    #                             param_distributions=params,
    #                             scoring='accuracy',
    #                             cv=5, verbose=1, n_jobs=-1)

    mod = rs.fit(X_train_scale, y_train)

    pickle.dump(mod, open(f'Models/{model_name}.sav', 'wb'))

    print('Best params:', rs.best_params_)
    print('Train score: %.3f' % rs.best_score_)
    print('Test score: %.3f' % rs.score(X_test_scale, y_test))

    return mod
