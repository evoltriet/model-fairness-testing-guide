# Source: https://github.com/valeria-io/bias-in-credit-models

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from typing import Tuple

def split_test_set_by_binary_category(test_df: pd.DataFrame, binary_category_name: str, binary_categories: list) ->\
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a test dataframe into two, based on the binary category it belongs to (e.g. female/male).
    :param test_df: test dataframe used by XGBoost model with all loan details
    :param binary_category_name: the categorical column used ofr the split
    :param binary_categories: the binary categories used for the split
    :return: two dataframes filtered based on the binary categories
    """
    category_name_zero, category_name_one = binary_categories[0], binary_categories[1]

    test_zero = test_df[test_df[binary_category_name] == category_name_zero]
    test_one = test_df[test_df[binary_category_name] == category_name_one]

    return test_zero, test_one

def calculate_classification_metrics(actuals: pd.Series, predicted_proba: pd.Series, threshold: float) -> \
        Tuple[int, int, float, float, float, float]:
    """
    Calculates the metrics for a confusion matrix and the values for the true positive rate and false positive rate
    :param actuals: actual values for the target variable 'Defaulted'
    :param predicted_proba: predicted probability for the target variable
    :param threshold: the threshold used to classify the probability as defaulted or not
    :return: values for (1) the number of true positives (predicted to pay back and pays back), (2) the number of false
    positives (predicted to pay back, but defaults) as well as the aggregate values for (3) the positive rate, (4) the
    negative rate, (5) the true positive rate and (6) the false positive rate
    """

    predicted = predicted_proba.apply(lambda x: 1 if x >= threshold else 0)

    tn, fp, fn, tp = confusion_matrix(actuals, predicted).ravel()

    """ Positive rate: % classified as positive (% predicted to pay back a loan) """
    pr = (tp + fp) / (tn + fp + fn + tp)

    """ Negative rate: % classified as negative (% predicted to default) """
    nr = (tn + fn) / (tn + fp + fn + tp)

    """ True positive rate: % of all positive that were classified correctly to pay back a loan """
    tpr = tp / (tp + fn)

    """ False positive rate: % of all negatives that we miss-classified as being able to pay back a loan """
    fpr = fp / (fp + tn)

    return tp, fp, pr, nr, tpr, fpr


def run_algorithmic_interventions_df(df_dict, col_names_dict, weights_dict):
    results_columns = ['IntervationName', 'Profit', 'threshold_0', 'threshold_1', 'TruePositive0', 'FalsePositive0',
                       'PositiveRate0', 'NegativeRate0', 'TruePositiveRate0', 'FalsePositiveRate0', 'TruePositive1',
                       'FalsePositive1', 'PositiveRate1', 'NegativeRate1', 'TruePositiveRate1', 'FalsePositiveRate1']

    results_df = pd.DataFrame(columns=results_columns)

    thresholds = np.arange(0, 1.01, 0.01)

    df_0 = df_dict['group_0']
    df_1 = df_dict['group_1']

    for t0 in thresholds:
        for t1 in thresholds:

            results_group_0 = calculate_classification_metrics(df_0[col_names_dict['actuals_col_name']],
                                                               df_0[col_names_dict['predicted_col_name']], t0)

            results_group_1 = calculate_classification_metrics(df_1[col_names_dict['actuals_col_name']],
                                                               df_1[col_names_dict['predicted_col_name']], t1)

            tp0, fp0, pr0, nr0, tpr0, fpr0 = results_group_0
            tp1, fp1, pr1, nr1, tpr1, fpr1 = results_group_1

            profit_function = weights_dict['weight_tp'] * (tp0 + tp1) - weights_dict['weight_fp'] * (fp0 + fp1)

            """
            Intervention 1: Maximise profit - uses different or equal thresholds for each category without any
            constrains
            """
            results_df = results_df.append(
                pd.DataFrame(
                    columns=results_columns,
                    data=[('MaxProfit', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 2: Group unawareness - uses equal threshold for both categories without any constrains
            """
            if t0 == t1:
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('GroupUnawareness', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 3: Demographic parity - uses different or equal threshold for each category as soon as each
            group gets granted the same percentage of loans (equal positive rate)
            """
            if round(pr0, 2) == round(pr1, 2):
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('DemographicParity', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 4: Equal Opportunity - uses different or equal thresholds for each category as soon as each
            group has the same rate of correctly classified loans as paid (equal TPR)
            """
            if round(tpr0, 2) == round(tpr1, 2):
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('EqualOpportunity', profit_function, t0, t1) + results_group_0 + results_group_1]))

                """
                Intervention 5: Equalised Odds - uses different or equal thresholds for each category as soon as each
                group has the same rate of correctly classified loans as paid (equal TPR) AND each group has the same
                miss-classification rate of loans granted (equal FPR).
                """
                if round(fpr0, 2) == round(fpr1, 2):
                    results_df = results_df.append(pd.DataFrame(
                        columns=results_columns,
                        data=[('EqualisedOdds', profit_function, t0, t1) + results_group_0 + results_group_1]))

    return results_df


