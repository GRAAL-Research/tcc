# -*- coding: utf-8 -*-

import codecs
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import numpy as np
import pandas as pd
import regex as re
import time
from typing import Union, Generator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

from TextComplexityComputer import TextComplexityComputer


def list_all_texts() -> Generator[str, None, None]:
    """Liste les textes autres que les références du repertoire text_samples/

    Yields:
        string: Le nom des fichier
    """
    for file in os.scandir('text_samples/'):
        if file.name[0] != '_':
            yield file.name


def clean_texts(files: Union[str, list] = 'all'):
    """Function to format one or more texts contained in the 'text_samples/' directory

    Actions:
        - remove the characters: '_', '•', '○', '…' et '[…]';
        - removes alphanumeric bulleted lists of the form : '\d.', '\d)', '(\d)', '(\d.\d)', '\w)' and '(\w)' (where
    \d = numerical and \w is alphanumerical);
        - add spaces after the points if not present;
        - and removes any form of space succession (space, carriage return, tab, etc.) with a single space.

    Args:
        files (Union[str, list], optional): in the case of string type, 'files' contains the single file to be cleaned,
    in the form '{name}.txt' OR equals 'all' and the function will process all files contained in the directory
    text_samples/. In the case of the list, it contains a list of files to clean, in the form
    ['{name1}.txt, {name2}.txt,..., {nameN}.txt'] for a list of N elements. (default at 'all')
    """

    def mr_clean(file):
        """Make the changes

        Args:
            file (string): name of file to process, must be in the format: '{nom}.txt'
        """
        encoding = 'utf-8'
        with codecs.open('./text_samples/' + file, 'r', encoding) as f:
            text = f.read()
            text = text.replace('_', ' ')
            text = text.replace('•', ' ')
            text = text.replace('○', ' ')
            text = re.sub(r'\[…\]', ' ', text)
            text = text.replace('…', ' . ')
            text = re.sub(r'- \d+ -', ' ', text)
            text = re.sub(r'\d+\)?\. ', ' ', text)
            text = re.sub(r'\(?\d+(\.\d+)*\)?\.? ', ' ', text)
            text = re.sub(r'\(?\w+\) ', ' ', text)
            text = re.sub(r'\.+(\w+)', '. \g<1>', text)
            text = re.sub(r'\s+', ' ', text)
        print(text)
        inp = 'qq'
        while inp != 'y' and inp != 'n' and inp != '':
            inp = input('Save ? ([y]/n) : ').lower()
        if inp == 'y' or inp == '':
            with codecs.open('./text_samples/' + file, 'w+', encoding) as f:
                f.write(text)
            print('Saved')
        else:
            print('Aborted')

    if isinstance(files, str):
        if files == 'all':
            for fichier in list_all_texts():
                print('\n####', fichier, '####')
                mr_clean(fichier)
        else:
            mr_clean(files)
    elif isinstance(files, list):
        for fichier in files:
            print('\n####', fichier, '####')
            mr_clean(fichier)


def create_analysis_df(with_scaler: bool = True, df_name: str = "analysis_result"):
    """Créateur du pd.DataFrame analysis_df
    Ce DataFrame va permettre de stocker les valeurs des métriques pour tous les textes 
    contenues dans le repretoire text_samples/.

    Il sera utilisé par la suite dans les fonctions boxplot_metrics() et train()

    Ce pd.DataFrame est stocké au nom de 'analysis_result.pickle' dans le repertoire code/
    """
    if not with_scaler:
        tcc = TextComplexityComputer(scaler=None)
        with_scaler = ''
    else:
        tcc = TextComplexityComputer()
        with_scaler = 'scaled_'

    print('\nStarting analysis\n')
    analysis_df = pd.DataFrame({'Fichier': list(), 'Extraits': list(), 'Type de doc.': list()})

    starting_time = time.time()
    references_csv = pd.read_csv('text_samples/_reference.csv')

    size = len(references_csv)
    for i in range(size):
        fichier, type_doc, extrait = references_csv['Nom du fichier'][i], references_csv['Type de doc.'][i], \
                                     references_csv['Extraits'][i]

        with codecs.open('./text_samples/' + fichier + '.txt', 'r', 'utf-8') as tf:
            start_time = time.time()
            text = tf.read()
            results = pd.DataFrame([[fichier, extrait, type_doc]], columns=['Fichier', 'Extraits', 'Type de doc.'])
            scores = tcc.get_metrics_scores(text)
            results = pd.concat([results, scores], axis=1)
            analysis_df = analysis_df.append(results, ignore_index=True)
            print('{}/{:<5}\t {:<40}\t in {} seconds'.format(i + 1, size, fichier, time.time() - start_time))

    print('in', time.time() - starting_time, 'seconds')

    print(analysis_df)
    inp = 'qq'
    while inp != 'y' and inp != 'n' and inp != '':
        inp = input('Save ? ([y]/n) : ').lower()
    if inp == 'y' or inp == '':
        with open('out_pickle/' + with_scaler + df_name + '.pickle', 'wb+') as f:
            pickle.dump(analysis_df, f)
        print('Saved')
    else:
        print('Aborted')


def boxplot_metrics(sorted_by: str = None):
    """Displays a box plot, based on the results of the metrics on the text samples

    /!\ This function is based on the pd.DataFrame 'scaled_analysis_result.pickle' /!\
    It must first be generated using the create_analysis_df() function

    Args:
        sorted_by (string, optional): sort the document types according to a metric.
    (by default empty, so sorted alphabetically).
    """
    with open('dfs/scaled_classified_df.pickle', 'rb') as rb:
        classified_df = pickle.load(rb)
    # if sorted_by:

    print(classified_df.index[0])
    fig, axes = plt.subplots(8, 8, constrained_layout=True)
    fig.set_figwidth(38)
    fig.set_figheight(26)
    analysis_val = classified_df.drop(['Fichier', 'Extraits'], axis=1)
    gen = analysis_val.drop('Type de doc.', axis=1)
    trad = {'Lois': "Laws", 'Assurance': "Insur.", 'Dictée': "Dict.", 'Roman': "Nov.", 'Wikipédia': "Wiki.",
            'Article': "News", 'Recette': "Rece.",
            'Histoire': 'Stor.'}
    i = 0
    j = 0
    for pkey in gen.keys():
        line = list()
        xticks = list()
        for fkey, group in analysis_val.sort_values(by="Difficulty").groupby('Type de doc.', sort=False)[pkey]:
            line.append(group)
            xticks.append(trad[fkey])
        # sns.boxplot(ax=axes[i][j], data=line, x='Doc. type', y='Score')
        axes[i][j].boxplot(line)
        axes[i][j].set_title(pkey)
        axes[i][j].set_xticklabels(xticks, rotation=45)
        axes[i][j].grid()

        if i == 7:
            i = 0
            j += 1
        else:
            i += 1

    # with PdfPages(r'/home/le-smog/Desktop/Partage/Charts.pdf') as export_pdf:
    #     export_pdf.savefig()
    plt.show()


def export_boxplot_metrics(sorted_by: str = None):
    """Displays a box plot, based on the results of the metrics on the text samples

    /!\ This function is based on the pd.DataFrame 'scaled_analysis_result.pickle' /!\
    It must first be generated using the create_analysis_df() function

    Args:
        sorted_by (string, optional): sort the document types according to a metric.
    (by default empty, so sorted alphabetically).
    """
    with open('dfs/scaled_classified_df.pickle', 'rb') as rb:
        classified_df = pickle.load(rb)
    # if sorted_by:

    analysis_val = classified_df.drop(['Fichier', 'Extraits'], axis=1)
    gen = analysis_val.drop('Type de doc.', axis=1)
    trad = {'Lois': "Laws", 'Assurance': "Insur.", 'Dictée': "Dict.", 'Roman': "Nov.", 'Wikipédia': "Wiki.",
            'Article': "News", 'Recette': "Rece.",
            'Histoire': 'Stor.'}
    i = 0
    j = 0
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(2)
    for pkey in gen.keys():
        line = list()
        xticks = list()
        for fkey, group in analysis_val.sort_values(by="Difficulty").groupby('Type de doc.', sort=False)[pkey]:
            line.append(group)
            xticks.append(trad[fkey])
        # sns.boxplot(ax=axes[i][j], data=line, x='Doc. type', y='Score')
        plt.boxplot(line)
        plt.xticks(np.arange(9), [''] + xticks, rotation=45)
        plt.axhline(0, color='black')
        plt.savefig('out_data/imgs/boxplot_' + pkey + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.clf()

        if i == 7:
            i = 0
            j += 1
        else:
            i += 1

    # with PdfPages(r'/home/le-smog/Desktop/Partage/Charts.pdf') as export_pdf:
    #     export_pdf.savefig()
    plt.show()


def create_scaler_save(df=pd.DataFrame(), scaler=None):
    if df.empty:
        with open('dfs/analysis_result.pickle', 'rb') as re:
            df = pickle.load(re)
    X_raw = df.drop(['Fichier', 'Extraits', 'Type de doc.'], axis=1)
    if not scaler:
        sc = StandardScaler()
        mm = MinMaxScaler()
        sc.fit(X_raw)
        mm.fit(X_raw)
        with open("out_pickle/StandardScaler.pickle", "wb+") as w:
            pickle.dump(sc, w)
        with open("out_pickle/MinMaxScaler.pickle", "wb+") as w:
            pickle.dump(mm, w)


if __name__ == "__main__":
    # clean_texts()
    # create_analysis_df(with_scaler=False)
    # create_scaler_save()
    # create_analysis_df()
    # if not os.path.exists('analysis_result.pickle'):
    #     create_analysis_df()
    # boxplot_metrics()
    # export_boxplot_metrics()
    with open('dfs/scaled_classified_df.pickle', 'rb') as re:
        df = pickle.load(re)
    print(df.head())
