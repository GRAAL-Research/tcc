# -*- coding: utf-8 -*-
import os
import pickle
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, plot_roc_curve, \
    plot_precision_recall_curve
from scipy.cluster.hierarchy import dendrogram
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from TextComplexityComputer import TextComplexityComputer
from IPython.display import display

from model_training.metrics_analysis import create_analysis_df, create_scaler_save


def get_Xy(df: pd.DataFrame, scale_with: callable = None):
    X, y = df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Difficulty'], axis=1), df['Difficulty']

    # if df.columns[2] == 'Type de doc.':
    #     score = {'Lois': 7, 'Assurance': 6, 'Wikipédia': 5, 'Roman': 4, 'Article': 3, 'Dictée': 2, 'Recette': 1,
    #              'Histoire': 0}
    #     y = df['Type de doc.'].apply(lambda x: score[x])
    #     X = df.drop(['Fichier', 'Extraits', 'Type de doc.'], axis=1)
    # elif df.columns[2] == 'Classes':
    #     y = df['Classes']
    #     X = df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Classes'], axis=1)
    # else:
    #     X, y = pd.DataFrame(), pd.DataFrame()
    if scale_with:
        sc = scale_with()
        X = sc.fit_transform(X)

    return X, y


def get_tested_features(return_df=False, force_print=False):
    if not return_df:
        force_print = True

    with open('dfs/scaled_classified_df_old.pickle', 'rb') as rb:
        classified_df = pickle.load(rb)

    X, y = get_Xy(classified_df)

    models_supervised = [LogisticRegression(random_state=0, max_iter=10000), DecisionTreeClassifier(random_state=0)]
    unsupported_list = list()
    for model in models_supervised:
        select_model = SelectFromModel(model).fit(X, y)
        for i, support in enumerate(select_model.get_support()):
            if support:
                unsupported_list.append(i)

    keys = classified_df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Difficulty'], axis=1).keys()
    unsupported = list()
    for val in set(unsupported_list):
        if unsupported_list.count(val) == 2:
            unsupported.append(keys[val])
    unsupported = set(unsupported)
    if force_print:
        print(unsupported)
    if return_df:
        return classified_df.drop(unsupported, axis=1)


def _get_score(clf, X, y):
    print(type(clf).__name__)
    print('Negative RMSE:', cross_val_score(clf, X, y, cv=3, scoring='neg_root_mean_squared_error').mean())
    print('Accuracy:', cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean())
    # TODO : group_by(y) isoler les niveaux de difficultes et neg RMSE puis accurracy en cross_val_score
    print()


def train(classified_df: pd.DataFrame = pd.DataFrame(), prescaled=True):
    if classified_df.empty:
        if not prescaled:
            with open('dfs/classified_df.pickle', 'rb') as rb:
                classified_df = pickle.load(rb)
            X, y = get_Xy(classified_df, scale_with=StandardScaler)
        else:
            with open('dfs/scaled_classified_df.pickle', 'rb') as rb:
                classified_df = pickle.load(rb)
            X, y = get_Xy(classified_df)

    class BaselineModel:
        def __init__(self):
            self.metrics = pd.DataFrame()

        def fit(self, X, y):
            pass

        def get_params(self, **kwargs):
            return dict()

        def predict(self, X):
            results = list()
            for km_score in X['km_score']:
                res = float(km_score) * 40 + 60
                if res < 30:
                    results.append(7)
                elif res < 40:
                    results.append(6)
                elif res < 50:
                    results.append(5)
                elif res < 60:
                    results.append(4)
                elif res < 70:
                    results.append(3)
                elif res < 80:
                    results.append(2)
                elif res < 90:
                    results.append(1)
                else:
                    results.append(0)
            return results

    bl = BaselineModel()
    _get_score(bl, X, y)

    param_grid1 = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15), 'random_state': [42]}
    model1 = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid1, cv=3,
                          scoring=("neg_mean_squared_error", "accuracy"), refit="neg_mean_squared_error").fit(X,
                                                                                                              y)
    index = np.where(model1.cv_results_['rank_test_neg_mean_squared_error'] == 1)[0][0]
    print(type(model1.best_estimator_).__name__)
    print(model1.cv_results_['mean_test_accuracy'][index])
    print(model1.cv_results_['mean_test_neg_mean_squared_error'][index])
    print()

    param_grid2 = {'penalty': ['l2'],
                   'C': np.logspace(-4, 4, 20),
                   'solver': ['lbfgs', 'newton-cg'], 'multi_class': ['ovr', 'multinomial', 'auto'],
                   'random_state': [42]}
    model2 = GridSearchCV(LogisticRegression(max_iter=10000), param_grid=param_grid2, cv=3,
                          scoring=("neg_mean_squared_error", "accuracy"), refit="neg_mean_squared_error").fit(X,
                                                                                                              y)
    index = np.where(model2.cv_results_['rank_test_neg_mean_squared_error'] == 1)[0][0]
    print(type(model2.best_estimator_).__name__)
    print(model2.cv_results_['mean_test_accuracy'][index])
    print(model2.cv_results_['mean_test_neg_mean_squared_error'][index])
    print()

    param_grid2a = {'penalty': ['l1', 'l2'],
                    'C': np.logspace(-4, 4, 20), 'solver': ['liblinear'], 'multi_class': ['ovr', 'auto'],
                    'random_state': [42]}

    model2a = GridSearchCV(LogisticRegression(max_iter=10000), param_grid=param_grid2a, cv=3,
                           scoring=("neg_mean_squared_error", "accuracy"), refit="neg_mean_squared_error").fit(X,
                                                                                                               y)
    index = np.where(model2a.cv_results_['rank_test_neg_mean_squared_error'] == 1)[0][0]
    print(type(model2a.best_estimator_).__name__)
    print(model2a.cv_results_['mean_test_accuracy'][index])
    print(model2a.cv_results_['mean_test_neg_mean_squared_error'][index])
    print()

    X_nb, y = get_Xy(classified_df, MinMaxScaler)

    param_grid3 = {
        'alpha': np.logspace(0, 10, 100),
        'fit_prior': [True, False]
    }

    model3 = GridSearchCV(MultinomialNB(), param_grid=param_grid3, cv=3,
                          scoring=("neg_mean_squared_error", "accuracy"), refit="neg_mean_squared_error").fit(X_nb, y)
    index = np.where(model3.cv_results_['rank_test_neg_mean_squared_error'] == 1)[0][0]
    print(type(model3.best_estimator_).__name__)
    print(model3.cv_results_['mean_test_accuracy'][index])
    print(model3.cv_results_['mean_test_neg_mean_squared_error'][index])
    print()

    # df = pd.DataFrame(model3.best_estimator_.feature_log_prob_,
    #                   columns=classified_df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Difficulty'],
    #                                              axis=1).keys(),
    #                   index=['Stor.', "Rece.", "News", "Wiki.", "Nov.", "Dict.", "Insur.", "Laws"])
    # df.to_excel("out_data/out_mnb.xlsx")

    # df = pd.Series(model1.best_estimator_.feature_importances_,
    #                index=classified_df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Difficulty'],
    #                                         axis=1).keys())
    # pd.set_option('display.max_rows', None)
    # print(df.sort_values())

    # fig = plt.figure(figsize=(25, 20))
    # _ = tree.plot_tree(model1.best_estimator_,
    #                    feature_names=classified_df.drop(['Fichier', 'Extraits', 'Type de doc.', 'Difficulty'],
    #                                                     axis=1).keys(),
    #                    class_names=['Stor.', "Rece.", "News", "Wiki.", "Nov.", "Dict.", "Insur.", "Laws"], filled=True)
    # plt.savefig('Tree.pdf', bbox_inches='tight', pad_inches=0)

    inp = 'qq'
    while inp != 'y' and inp != 'n' and inp != '':
        inp = input('Save ? ([y]/n) : ').lower()
    if inp == 'y' or inp == '':
        with open('out_pickle/model.pickle', 'wb+') as f:
            pickle.dump(model3, f)
        print('Saved')
    else:
        print('Aborted')


def validate_model():
    with open("out_pickle/model.pickle", "rb") as m:
        model = pickle.load(m)

    tcc = TextComplexityComputer()
    for file in os.listdir("test_set"):
        with open("test_set/" + file, "r") as f:
            text = f.read()
            print(file, "--->", model.predict(tcc.get_metrics_scores(text)))


def find_difficulties_levels(analysis_df: pd.DataFrame = pd.DataFrame(), n_clusters=8, return_df=False,
                             force_print=False) -> pd.DataFrame:
    if not return_df:
        force_print = True
    if analysis_df.empty:
        with open('dfs/analysis_result.pickle', 'rb') as re:
            analysis_df = pickle.load(re)

    X, y = get_Xy(analysis_df)

    clf = KMeans(n_clusters)

    res = clf.fit_predict(X)

    df = pd.DataFrame(zip(res, analysis_df['Extraits']), columns=['Groupe', 'Extraits'])

    if force_print:
        for key, val in df.groupby('Groupe'):
            print('Class {}:\n{}\n\n'.format(key + 1, val))

        raw_df = analysis_df.drop(['Fichier', 'Extraits', 'Type de doc.'], axis=1)
        keys = raw_df.keys()
        print('Cluster centers :\n', pd.DataFrame(clf.cluster_centers_, columns=keys))
    if return_df:
        return df


def create_classified_df(with_scaler: bool = True, df_out: str = "classified_df", df_in: str = "analysis_result"):
    if with_scaler:
        with open('scaled_' + df_in + '.pickle', 'rb') as re:
            analysis_df = pickle.load(re)
        with_scaler = 'scaled_'
    else:
        with open(df_in + '.pickle', 'rb') as re:
            analysis_df = pickle.load(re)
        with_scaler = ''

    print(analysis_df.head())
    score = {'Lois': 7, 'Assurance': 6, 'Wikipédia': 5, 'Article': 4, 'Dictée': 3, 'Roman': 2, 'Recette': 1,
             'Histoire': 0}
    difficulty_list = [score[diff] for diff in analysis_df['Type de doc.']]

    analysis_df['Difficulty'] = difficulty_list

    print(analysis_df)
    inp = 'qq'
    while inp != 'y' and inp != 'n' and inp != '':
        inp = input('Save ? ([y]/n) : ').lower()
    if inp == 'y' or inp == '':
        with open(with_scaler + df_out + '.pickle', 'wb+') as f:
            pickle.dump(analysis_df, f)
        print('Saved')
    else:
        print('Aborted')


def plot_difficulties_levels(analysis_df: pd.DataFrame = pd.DataFrame()):
    if analysis_df.empty:
        with open('dfs/analysis_result.pickle', 'rb') as re:
            analysis_df = pickle.load(re)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    print(analysis_df[['mls', 'ps_30', 'nws_90', 'mlt', 'tu_s', 'ctu_tu', 'dc_c', 'c_s', 'c_tu', 'cp_c', 'cp_tu', 'pa',
                       'nlm', 'uni_gram_lem', 'msttr', 'mattr', 'mtld', 'fk_ease', 'bingui', 'km_score']])
    X, y = get_Xy(
        analysis_df[
            ['Fichier', 'Extraits', 'Type de doc.', 'mls', 'ps_30', 'nws_90', 'mlt', 'tu_s', 'ctu_tu', 'dc_c', 'c_s',
             'c_tu', 'cp_c', 'cp_tu', 'pa',
             'nlm', 'uni_gram_lem', 'msttr', 'mattr', 'mtld', 'fk_ease', 'bingui', 'km_score']]
    )
    model = model.fit(X)

    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    locs, labels = plt.xticks()
    new_xticks = list(map(lambda x: analysis_df['Extraits'][int(x.get_text())], labels))
    plt.xticks(locs, new_xticks, size=9)
    plt.tight_layout()
    plt.show()


def create_dfs():
    create_analysis_df(with_scaler=False)
    create_scaler_save()
    os.rename("StandardScaler.pickle", "../TextComplexityComputer/StandardScaler.pickle")
    create_analysis_df()
    create_classified_df()


def get_correlation():
    with open('dfs/scaled_classified_df.pickle', 'rb') as rb:
        classified_df = pickle.load(rb)
    pd.set_option('display.max_rows', None)
    print(classified_df.corr(method='spearman')['Difficulty'].dropna().abs().sort_values())


if __name__ == "__main__":
    # plot_difficulties_levels()
    # find_difficulties_levels()
    # create_classified_df(df_name="classified_df_WO_biberpy")
    # create_classified_df()
    # create_dfs()
    # get_correlation()
    # get_tested_features()
    train()  # classified_df=get_tested_features(return_df=True, force_print=True))
    # create_model()
    # validate_model()
