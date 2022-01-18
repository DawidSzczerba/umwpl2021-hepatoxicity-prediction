import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle


def get_dual_cv_results(hyperparams_grid, model, cv_outer, cv_inner, x, y):
    cv_results = pd.DataFrame()

    for train_index, test_index in cv_outer.split(x):
        # split data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # define search
        search = GridSearchCV(model, hyperparams_grid, scoring='r2', cv=cv_inner,
                              return_train_score=True, verbose=1000)
        search_fit = search.fit(x_train, y_train)
        cv_result = pd.DataFrame(search_fit.cv_results_)
        cv_results = cv_results.append(cv_result, ignore_index=True)
    return cv_results


def group_dual_cv_results_data(cv_results):
    group_keys = [value for value in cv_results.columns.values if
                  value.startswith("param_") and 'random_state' not in value]
    groups = cv_results.groupby(by=group_keys)
    agg_arguments = {'mean_test_score': 'mean', 'std_test_score': 'mean',
                     'mean_train_score': 'mean', 'std_train_score': 'mean'}
    return groups.agg(agg_arguments)


def get_best_dual_cv_results(grouped_cv_results, key='mean_test_score'):
    return grouped_cv_results[key].max()


def get_best_hyperparameters(grouped_cv_results, key='mean_test_score'):
    return grouped_cv_results[key].idxmax()


def calculate_model_results(model, cv_outer, x, y):
    results = pd.DataFrame(columns=['train_score', 'test_score', 'pred_train', 'pred_test'])

    for i, (train_index, test_index) in enumerate(cv_outer.split(x)):
        # split data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        # save the model to disk
        model_name = str(model).split("(", 1)[0]
        filename = f"../models/finalized_{model_name}_model_{i}.pickle"
        pickle.dump(model, open(filename, 'wb'))
        y_pred_train = model.predict(x_train)
        train_score = model.score(x_train, y_train)
        y_pred_test = model.predict(x_test)
        test_score = model.score(x_test, y_test)
        results.loc[i] = [train_score, test_score, y_pred_train, y_pred_test]
    return results


def get_final_model_result(results):
    return results['test_score'].mean()



