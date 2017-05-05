#Modified based on Rayid's magic loop

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import *
import random
import pylab as pl
import matplotlib.pyplot as plt
import time
import seaborn as sns

clfs = {'RF': RandomForestClassifier(),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
    'LR': LogisticRegression(),
    'SVM': SVC(probability=True, random_state=0),
    'DT': DecisionTreeClassifier(),
    'BAG': BaggingClassifier(),
    'KNN': KNeighborsClassifier() 
    }
    
small_grid = {'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
    'LR': {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10]},
    'SVM' :{'C' :[0.01,0.1,1,10],'kernel':['linear']},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
    'BAG':{'n_estimators    ':[1,10,20,50], 'max_samples':[5,10], 'max_features': [5,10]},
    'KNN' :{'n_neighbors': [1,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
    }

def extract_features(df, var_lst, d):
    '''
    Determine which independent variables
    to be used in the logit regression.

    Return dataframes of all available data
    for y and x's.
    '''
    y = df[d['y']]
    ind_vars = []
    for i in var_lst:
        ind_vars.append(d['x'+str(i)])
    X = df[ind_vars]
    return X, y

def generate_binary_at_k(y_scores, k):
    '''
    Set first k% as 1, the rest as 0.
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def scores_at_k(y_true, y_scores, k):
    '''
    For a given level of k, calculate corresponding
    precision, recall, and f1 scores.
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1


def plot_precision_recall(y_true, y_prob, model, p):
    '''
    Plots the PR curve given true value and predicted
    probilities of y.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)    
    plt.clf()
    plt.plot(recall, precision, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for {} model: AUC={:.2f} \n with parameters: {}'.\
        format(model, average_precision_score(y_true, y_prob), p))
    plt.legend(loc="lower left")
    plt.show()


def clf_loop(models_to_run, X, y):
    '''
    Given the classifiers to test, run with parameters from the small_grid.
    Records metrics in a Dataframe:
        accuracy, AUC of ROC curve and PR curve,
        time used,
        precision, recall, and f1 scores at k = 5, 10, or 20.
    '''
    results_df = pd.DataFrame(columns=('model_type','parameters', 'accuracy','auc-roc', 'auc-pr', 'time', 'precision,recall,f1 at_5',
     'precision,recall,f1 at_10', 'precision,recall,f1 at_20'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = small_grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                start = time.time()
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                accuracy = clf.score(X_test, y_test)
                end = time.time()

                #Zip, unzip to ensure corresponding order
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

                results_df.loc[len(results_df)] = [models_to_run[index], p, accuracy,
                                                    roc_auc_score(y_test, y_pred_probs),
                                                    average_precision_score(y_test, y_pred_probs), end-start,
                                                    scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                    scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                    scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]

                plot_precision_recall(y_test,y_pred_probs,models_to_run[index],p)

            except IndexError as e:
                print('Error:',e)
                continue
    return results_df