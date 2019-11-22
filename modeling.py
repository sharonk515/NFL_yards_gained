import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

def run_randomized_search(model, params, X_train, X_test,
                          y_train, y_test, random_state=None):

    '''
    This function runs various models.
    Parameters
    ----------
    model:
    params:
    X_train:
    X_test:
    y_train:
    y_test:
    random_state:

    Returns
    --------
    Model with the best hyperparameters, train score, and test score.
    '''
    rs = RandomizedSearchCV(estimator=model(random_state=random_state),
                            param_distributions=params,
                            scoring='r2',
                            cv=5, verbose=1, n_jobs=-1)

    mod = rs.fit(X_train, y_train)

    print('Best params:', rs.best_params_)
    print('Train score: %.3f' % rs.best_score_)
    print('Test score: %.3f' % rs.score(X_test, y_test))

    return mod


def run_randomized_search_svr(params, X_train_scale, X_test_scale,
                              y_train, y_test):
    '''
    This function runs Support Vector Regression with standardized data.
    Parameters
    ----------
    model:
    params:
    X_train_scale:
    X_test_scale:
    y_train:
    y_test:
    random_state:

    Returns
    --------
    Model with the best hyperparameters, train score, and test score.
    '''
    rs = RandomizedSearchCV(estimator=SVR(),
                            param_distributions=params,
                            scoring='r2',
                            cv=5, verbose=1, n_jobs=-1)

    mod = rs.fit(X_train_scale, y_train)

    print('Best params:', rs.best_params_)
    print('Train score: %.3f' % rs.best_score_)
    print('Test score: %.3f' % rs.score(X_test_scale, y_test))

    return mod
