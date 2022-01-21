import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lime import lime_tabular
import pickle


def draw_actual_vs_predicted_plot(folder_path, number, y_test, y_pred_test, result):
    x = np.arange(start=0, stop=len(y_test), step=1)
    y_test = np.expm1(y_test)
    y_pred_test = np.expm1(y_pred_test)
    fontsize = 15
    plt.figure(figsize=(14, 6))
    plt.plot(x, y_test, 'go', label="ALT")
    plt.plot(x, y_pred_test, 'bD', label="ALT - predicted")
    plt.title(f"PREDICTED VS ACTUAL", fontsize=fontsize)
    plt.suptitle(f"R2 ACCURACY = {result}", fontsize=fontsize)
    plt.xlabel("SAMPLE NUMBER", fontsize=fontsize)
    plt.ylabel("ALT  VALUE", fontsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize)
    plt.xticks(range(0, 20))
    plt.yticks(range(0, 300, 25))
    plt.grid()
    plt.savefig(f'{folder_path}/actual_vs_predicted_{number}.jpg')
    plt.show()


def model_explanation(folder_path, number, model, x_train, x_test, y_test, y_pred_test, features):
    """
    Function generates html reports using LIME framework (https://github.com/marcotcr/lime)
    for best and worst prediction
    """
    sorted_absolute_error_idx = np.argsort(np.absolute(np.subtract(y_test, y_pred_test)))
    explainer = lime_tabular.LimeTabularExplainer(x_train, mode="regression",
                                                  feature_names=features)
    explanation_best_predicted = explainer.explain_instance(x_test[sorted_absolute_error_idx[0]],
                                                            model.predict,
                                                            num_features=len(features))

    explanation_worst_predicted = explainer.explain_instance(
        x_test[sorted_absolute_error_idx[::-1][0]], model.predict,
        num_features=len(features))
    explanation_best_predicted.save_to_file(
        f"{folder_path}/explanation_best_prediction_{number}.html")
    explanation_worst_predicted.save_to_file(
        f"{folder_path}/explanation_worst_prediction_{number}.html")

    return explanation_best_predicted.as_list(), explanation_worst_predicted.as_list()



