import nltk
import pymorphy2
import random
import pandas as pd
import numpy as np
import sys
import os
import json
import requests
import tarfile
import itertools
from datetime import datetime
import matplotlib.pyplot as plt

from copy import deepcopy
from time import time
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from os.path import join, isdir, isfile
from os import mkdir

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

morph = pymorphy2.MorphAnalyzer()


def normal_time(z):
    if z > 1:
        h = z/3600
        m = z % 3600/60
        s = z % 3600 % 60
        t = '%i:%i:%i' % (h, m, s)
    else:
        t = str(z)
    return t


# -------------------------- Hyper search ----------------------------------


class HyperPar:
    def __init__(self, **kwargs):
        np.random.seed(int(time()))
        self.params = kwargs

    def sample_params(self):
        params = deepcopy(self.params)
        params_sample = dict()
        for param, param_val in params.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = bool(np.random.choice([True, False]))
                elif 'range' in param_val:
                    # Generate number of smaples
                    if 'n_samples' in param_val:
                        if param_val['n_samples'] > 1 and param_val.get('increasing', False):

                            sample_1 = self._sample_from_ranges(param_val)
                            sample_2 = self._sample_from_ranges(param_val)
                            start_stop = sorted([sample_1, sample_2])
                            sample = [s for s in np.linspace(start_stop[0], start_stop[1], param_val['n_samples'])]
                            if param_val.get('discrete', False):
                                sample = [int(s) for s in sample]
                        else:
                            sample = [self._sample_from_ranges(param_val) for _ in range(param_val['n_samples'])]
                    else:
                        sample = self._sample_from_ranges(param_val)
                params_sample[param] = sample
            else:
                params_sample[param] = param_val
        return params_sample

    def _sample_from_ranges(self, opts):
        from_ = opts['range'][0]
        to_ = opts['range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_, to_):
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)


class ConfGen(object):
    def __init__(self, config, search_config, seed=None):
        if seed is None:
            np.random.seed(int(time()))
        else:
            np.random.seed(seed)

        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            with open(config) as c:
                conf = json.load(c)
                c.close()
            self.config = conf
        else:
            raise ValueError('Input parameter {0} must be dict or path to json file'
                             'but {1} was found.'.format('config', type(config)))

        if isinstance(search_config, dict):
            self.params = search_config
        elif isinstance(search_config, str):
            with open(search_config) as c:
                conf = json.load(c)
                c.close()
            self.params = conf
        else:
            raise ValueError('Input parameter {0} must be dict or path to json file'
                             'but {1} was found.'.format('config', type(config)))

        for key in self.params.keys():
            if key not in self.config.keys():
                raise ValueError('Key {} is absent in config dict.'.format(key))

    def _sample_params(self):
        params = deepcopy(self.params)
        params_sample = dict()
        for param, param_val in params.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = bool(np.random.choice([True, False]))
                elif 'range' in param_val:
                    # Generate number of smaples
                    if 'n_samples' in param_val:
                        if param_val['n_samples'] > 1 and param_val.get('increasing', False):

                            sample_1 = self._sample_from_ranges(param_val)
                            sample_2 = self._sample_from_ranges(param_val)
                            start_stop = sorted([sample_1, sample_2])
                            sample = [s for s in np.linspace(start_stop[0], start_stop[1], param_val['n_samples'])]
                            if param_val.get('discrete', False):
                                sample = [int(s) for s in sample]
                        else:
                            sample = [self._sample_from_ranges(param_val) for _ in range(param_val['n_samples'])]
                    else:
                        sample = self._sample_from_ranges(param_val)
                params_sample[param] = sample
            else:
                params_sample[param] = param_val
        return params_sample

    def _sample_from_ranges(self, opts):
        from_ = opts['range'][0]
        to_ = opts['range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_, to_):
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)

    def sample_params(self):
        part = self._sample_params()
        for key in part.keys():
            self.config[key] = part[key]

        return self.config

    def generator(self, N):
        for i in range(N):
            part = self._sample_params()
            for key in part.keys():
                self.config[key] = part[key]

            yield self.config


# -------------------------- Common ----------------------------------


def get_result(y_pred, y_test):
    category_description = list(set(y_test))

    results = dict()
    results['accuracy'] = 'accuracy : {}'.format(accuracy_score(y_test, y_pred))
    results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    # results['ROC'] = roc_auc_score(y_test, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    results['classes'] = {}

    for i in range(len(category_description)):
        y_bin_pred = np.zeros(y_pred.shape)
        y_bin_pred[y_pred == i] = 1
        y_bin_answ = np.zeros(y_pred.shape)
        y_bin_answ[y_test == i] = 1

        precision_tmp = precision_score(y_bin_answ, y_bin_pred)
        recall_tmp = recall_score(y_bin_answ, y_bin_pred)
        if recall_tmp == 0 and precision_tmp == 0:
            f1_tmp = 0.
        else:
            f1_tmp = 2 * recall_tmp * precision_tmp / (precision_tmp + recall_tmp)

        results['classes'][str(category_description[i])] = {'number_test_objects': y_bin_answ[y_test == i].shape[0],
                                                            'precision': precision_tmp,
                                                            'recall': recall_tmp,
                                                            'f1': f1_tmp}

    # string_to_format = '{:7} number_test_objects: {:4}   precision: {:5.3}   recall: {:5.3}  f1: {:5.3}'
    # results['classes'].append(string_to_format.format(category_description[i],
    #                                                   y_bin_answ[y_test == i].shape[0],
    #                                                   precision_tmp,
    #                                                   recall_tmp,
    #                                                   f1_tmp))

    return results


# -------------------------- Logging ----------------------------------


def logging(res, pipe_conf, name, language='russian', dataset_name='vkusvill', root='/home/mks/projects/DeepBurtsev/'):
    log = {'pipeline configuration': pipe_conf, 'results': res}

    root = root
    path = join(root, 'results', language, dataset_name, name)

    if not os.path.isdir(path):
        os.makedirs(path)

    file = name + '.txt'

    if not isfile(join(path, file)):
        with open(join(path, file), 'w') as f:
            line = json.dumps(log)
            f.write(line)
            f.write('\n')
            f.close()
    else:
        with open(join(path, file), 'a') as f:
            line = json.dumps(log)
            f.write(line)
            f.write('\n')
            f.close()

    return None

# -------------------------- Results visualization ----------------------------------


def results_analizator(log, target_metric='f1_weighted', num_best=3):
    models_names = list(log['experiments'].keys())
    metrics = log['experiment_info']['metrics']
    if target_metric not in metrics:
        print("Warning: Target metric '{}' not in log. The fisrt metric in log will be use as target metric.".format(
            target_metric))
        target_metric = metrics[0]

    main = {}
    for name in models_names:
        main[name] = {}
        for met in metrics:
            main[name][met] = []

    for name in models_names:
        for key, val in log['experiments'][name].items():
            for met in metrics:
                main[name][met].append(val['results'][met])

    for name in models_names:
        for key, val in log['experiments'][name].items():
            for met in metrics:
                tmp = np.sort(main[name][met])[-num_best:]
                main[name][met] = tmp[::-1]

    m = 0
    mxname = ''
    for name in models_names:
        if main[name][target_metric][0] > m:
            m = main[name][target_metric][0]
            mxname = name

    main['best_model'] = mxname
    main['target_metric'] = target_metric
    main['best_score'] = m

    return main


def get_table(dang, savepath, root, filename='report', ext='pdf'):
    # make dataframe table
    fun_0 = lambda p: [dang[x][p] for x in dang.keys()]
    table = pd.DataFrame({'Models': list(dang.keys()),
                          'Accuracy': fun_0('max_acc'),
                          'F1 macro': fun_0('max_f1_macro'),
                          'F1 weighted': fun_0('max_f1_weighted')})

    table = pd.pivot_table(table, index='Models', values=['Accuracy', 'F1 macro', 'F1 weighted'], fill_value=0)

    # best model
    name_best_model = list(table[table['F1 weighted'] == table['F1 weighted'].max()].index)[0]
    I = dang[name_best_model]['index_of_best']
    best_model = {}
    a = dang[name_best_model]['list'][I]['results']['classes']
    for x in a.keys():
        for y in a[x].keys():
            if y not in best_model.keys():
                best_model[y] = list()
                best_model[y].append(a[x][y])
            else:
                best_model[y].append(a[x][y])

    # create pdf table
    print(root)
    env = Environment(loader=FileSystemLoader(root))
    template = env.get_template("./deepburtsev/core/template.html")
    template_vars = {"title": "Results ",
                     "national_pivot_table": table.to_html()}

    html_out = template.render(template_vars)

    adr = join(savepath, '{0}.{1}'.format(filename, ext))

    HTML(string=html_out).write_pdf(adr)

    return table, [name_best_model, best_model]


def plot_res_table(info, save=False, savepath='./', width=0.35, ext='png'):
    # prepeare data
    bar_list = []
    models = list(info.keys())
    metrics = list(info[models[0]].keys())
    n = len(metrics)

    for met in metrics:
        tmp = []
        for model in models:
            tmp.append(info[model][met][0])
        bar_list.append(tmp)

    x = np.arange(len(models))

    # ploting
    fig, ax = plt.subplots()

    colors = plt.cm.Paired(np.linspace(0, 0.5, len(bar_list)))

    bars = []
    for i, y in enumerate(bar_list):
        if i == 0:
            bars.append(ax.bar(x, y, width, color=colors[i]))
        else:
            bars.append(ax.bar(x + width, y, width, color=colors[i]))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n):
        cell_text.append(['test' for x in models])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=metrics,
                          rowColours=colors,
                          colLabels=models,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # plot x sticks and labels
    plt.xticks([])

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric')

    # plot legend
    ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics))

    # auto lables
    def autolabel(bars):
        """
        Attach a text label above each bar displaying its height
        """
        for rects in bars:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

    autolabel(bars)

    # show the picture
    if save:
        if not isdir(savepath):
            mkdir(savepath)
        adr = join(savepath, '{0}.{1}'.format('main_hist', ext))
        fig.savefig(adr, dpi=100)
        plt.close(fig)
    else:
        plt.show()
    return None


def plot_res(info, save=True, savepath='./', width=0.35, ext='png'):
    # prepeare data
    bar_list = []
    models = list(info.keys())
    metrics = list(info[models[0]].keys())
    n = len(metrics)

    for met in metrics:
        tmp = []
        for model in models:
            tmp.append(info[model][met][0])
        bar_list.append(tmp)

    x = np.arange(len(models))

    # ploting
    fig, ax = plt.subplots()

    colors = plt.cm.Paired(np.linspace(0, 0.5, len(bar_list)))
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric')

    bars = []
    for i, y in enumerate(bar_list):
        if i == 0:
            bars.append(ax.bar(x, y, width, color=colors[i]))
        else:
            bars.append(ax.bar(x + width, y, width, color=colors[i]))

    # plot x sticks and labels
    ax.set_xticks(ind - width / 2 + n * width / 2)
    ax.set_xticklabels(tuple(models))

    # plot legend
    ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics))

    # auto lables
    def autolabel(bars):
        """
        Attach a text label above each bar displaying its height
        """
        for rects in bars:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

    autolabel(bars)

    # show the picture
    if not save:
        plt.show()
    else:
        if not isdir(savepath):
            mkdir(savepath)
        adr = join(savepath, '{0}.{1}'.format('main_hist', ext))
        fig.savefig(adr, dpi=100)
        plt.close(fig)

    return None


def plot_confusion_matrix(matrix, important_categories, plot_name='confusion matrix',
                          plot_size=(30, 30), fontsize=16, ticks_size=10,
                          axis_names=['X', 'Y'], ext='png', savepath='./results/images/'):
    fig, ax = plt.subplots()
    fig.set_figwidth(plot_size[0])
    fig.set_figheight(plot_size[1])

    plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(plot_name, fontsize=fontsize)

    plt.xticks(np.arange(0, len(important_categories)), important_categories, rotation=90, fontsize=ticks_size)
    plt.yticks(np.arange(0, len(important_categories)), important_categories, fontsize=ticks_size)

    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(np.arange(matrix.shape[0]), np.arange(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel(axis_names[0], fontsize=14)
    plt.ylabel(axis_names[1], fontsize=14)

    adr = join(savepath, '{0}.{1}'.format(plot_name, ext))
    plt.savefig(adr)
    plt.close(fig)

    return None


######################################################################################

# # data
# N = 5
# ind = np.arange(N)
#
# men_means = (20, 35, 30, 35, 27)
# women_means = (25, 32, 34, 20, 25)
#
# info = {}
# for i, x in enumerate(ind):
#     info[str(x)] = {'Men': [men_means[i]], 'Women': [women_means[i]]}
# #     info[str(x)] = {'Men': [men_means[i]]}
#
# def results_visualization(root):
#     save_path = join(root, 'results')
#     with open(join(root, root.split('/')[-1] + '.json'), 'r') as log_file:
#         log = json.load(log_file)
#         log_file.close()
#
#     # reading and scrabbing data
#     info = results_analizator(log)
#     plot_res(info, save_path)
#
#     return None

######################################################################################


def results_summarization(root, date=None, language='russian', dataset_name='vkusvill'):
    path = join('./results/', language, dataset_name)

    if date is None:
        date = datetime.now()
        date_path = join(path, '{}-{}-{}'.format(date.year, date.month, date.day))
        if not isdir(date_path):
            os.makedirs(date_path)

        image_path = join(date_path, 'images')
        if not isdir(image_path):
            os.makedirs(image_path)

        log = join(date_path, '{}-{}-{}.txt'.format(date.year, date.month, date.day))
        if not isfile(log):
            # with open(log, 'w') as log_file:
            #     log_file.close()
            raise FileExistsError('File with results {}'
                                  ' is not exist'.format('{}-{}-{}.txt'.format(date.year, date.month, date.day)))

    else:
        date_path = join(path, '{}-{}-{}'.format(date.year, date.month, date.day))
        if not isdir(date_path):
            os.makedirs(date_path)

        image_path = join(date_path, 'images')
        if not isdir(image_path):
            os.makedirs(image_path)

        log = join(date_path, '{}-{}-{}.txt'.format(date.year, date.month, date.day))

        if not isfile(log):
            # with open(log, 'w') as log_file:
            #     log_file.close()
            raise FileExistsError('File with results {} is not exist'.format('{}-{}-{}.txt'.format(date.year,
                                                                                                   date.month,
                                                                                                   date.day)))

    # reading and scrabbing data
    info = scrab_data(log)

    # make dataframe table
    table, best_model = get_table(info, date_path, root)

    # ploting results
    model_names = tuple(table.index)
    metrics = list(table.keys())
    x = np.arange(len(table))
    for i in metrics:
        y = list(table[i])
        axes_names = ['Models', i]

        ploting_hist(x, y, plot_name=i, axes_names=axes_names, x_lables=model_names, savepath=image_path)

    for n in model_names:
        I = info[n]['index_of_best']
        important_categories = list(info[n]['list'][I]['results']['classes'].keys())
        important_categories = np.array([int(x) for x in important_categories])
        matrix = np.array(info[n]['list'][I]['results']['confusion_matrix'])

        plot_confusion_matrix(matrix, important_categories,
                              plot_name='Confusion Matrix of {}'.format(n),
                              axis_names=['Prediction label', 'True label'],
                              savepath=image_path)

    best_model_name, stat = best_model
    classes_names = list(info[model_names[0]]['list'][0]['results']['classes'].keys())
    for i in stat.keys():
        axes_names = ['Classes', i]
        ploting_hist(np.arange(len(stat[i])), stat[i], plot_name=i, axes_names=axes_names, x_lables=classes_names,
                     savepath=image_path)

    return None


# -------------------------- Utils ----------------------------------


class NLTKTokenizer(object):

    def infer(self, batch, tokenizer="wordpunct_tokenize"):
        tokenized_batch = []

        tokenizer_ = getattr(nltk.tokenize, tokenizer, None)
        if callable(tokenizer_):
            if type(batch) == str:
                tokenized_batch = " ".join(tokenizer_(batch))
            else:
                # list of str
                for text in batch:
                    tokenized_batch.append(" ".join(tokenizer_(text)))
            return tokenized_batch
        else:
            raise AttributeError("Tokenizer %s is not defined in nltk.tokenizer" % tokenizer)


def tokenize(data):
    tok_data = list()
    for x in data:
        sent_toks = nltk.sent_tokenize(x)
        word_toks = [nltk.word_tokenize(el) for el in sent_toks]
        tokens = [val for sublist in word_toks for val in sublist]
        tok_data.append(tokens)

    return tok_data


def transform(data, lower=True, lemma=True, ngramm=False):
    Tokens = list()
    for x in tqdm(data['request']):
        sent_toks = nltk.sent_tokenize(x)
        word_toks = [nltk.word_tokenize(el) for el in sent_toks]
        tokens = [val for sublist in word_toks for val in sublist]
        tokens = [el for el in tokens if el != '']
        if lower:
            tokens = [el.lower() for el in tokens]
        if lemma:
            tokens = [morph.parse(el)[0].normal_form for el in tokens]
        if ngramm:
            bigram = list(nltk.bigrams(tokens))
            bigram = ['_'.join(x) for x in bigram]
            tokens.extend(bigram)

        Tokens.append(' '.join(tokens))

    df = pd.DataFrame({'request': Tokens,
                       'class': data['class']})
    return df


def split(data, prop):
    dd = dict()
    cd = dict()
    train = list()
    valid = list()
    test = list()

    for x, y in zip(data['request'], data['class']):
        if y not in dd.keys():
            dd[y] = list()
            cd[y] = 0
            dd[y].append((x, y))
            cd[y] += 1
        else:
            dd[y].append((x, y))
            cd[y] += 1

    if type(prop) is list:
        assert len(prop) == 2
        assert type(prop[0]) is float

        valid_ = dict()
        test_ = dict()

        for x in dd.keys():
            num = int(cd[x] * prop[0])
            valid_[x] = random.sample(dd[x], num)
            [dd[x].remove(t) for t in valid_[x]]

        for x in dd.keys():
            num = int(cd[x] * prop[1])
            test_[x] = random.sample(dd[x], num)
            [dd[x].remove(t) for t in test_[x]]
    else:
        raise ValueError('Split proportion must be list of floats, with length = 2')

    train_ = dd

    for x in train_.keys():
        for z_, z in zip([train_, valid_, test_], [train, valid, test]):
            z.extend(z_[x])

    del train_, valid_, test_, dd, cd

    for z in [train, valid, test]:
        z = random.shuffle(z)

    utrain, uvalid, utest, ctrain, cvalid, ctest = list(), list(), list(), list(), list(), list()
    for z, n, c in zip([train, valid, test], [utrain, uvalid, utest], [ctrain, cvalid, ctest]):
        for x in z:
            n.append(x[0])
            c.append(x[1])

    train = pd.DataFrame({'request': utrain,
                          'class': ctrain})
    valid = pd.DataFrame({'request': uvalid,
                          'class': cvalid})
    test = pd.DataFrame({'request': utest,
                         'class': ctest})

    return train, valid, test


def labels2onehot(labels, classes):
    n_classes = len(classes)
    eye = np.eye(n_classes)
    y = []
    for sample in labels:
        curr = np.zeros(n_classes)
        for intent in sample:
            if intent not in classes:
                # print('Warning: unknown class {} detected'.format(intent))
                curr += eye[np.where(classes == 'unknown')[0]].reshape(-1)
            else:
                curr += eye[np.where(classes == intent)[0]].reshape(-1)
        y.append(curr)
    y = np.asarray(y)
    return y


def labels2onehot_one(labels, classes, batch_size):
    n_classes = classes
    eye = np.eye(n_classes)
    y = np.zeros((batch_size, n_classes))
    for i, sample in enumerate(labels):
        y[i] = eye[sample-1]
    return y


def proba2labels(proba, confident_threshold, classes):
    y = []
    for sample in proba:
        to_add = np.where(sample > confident_threshold)[0]
        if len(to_add) > 0:
            y.append(classes[to_add])
        else:
            y.append([classes[np.argmax(sample)]])
    y = np.asarray(y)
    return y


def proba2onehot(proba, confident_threshold, classes):
    return labels2onehot(proba2labels(proba, confident_threshold, classes), classes)


def log_metrics(names, values, updates=None, mode='train'):
    sys.stdout.write("\r")  # back to previous line
    print("{} -->\t".format(mode), end="")
    if updates is not None:
        print("updates: {}\t".format(updates), end="")

    for id in range(len(names)):
        print("{}: {}\t".format(names[id], values[id]), end="")
    print(" ")  # , end='\r')
    return

# -------------------------- File Utils ----------------------------------


def download(dest_file_path, source_url):
    r""""Simple http file downloader"""
    print('Downloading from {} to {}'.format(source_url, dest_file_path))
    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)
    os.makedirs(datapath, mode=0o755, exist_ok=True)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def untar(file_path, extract_folder=None):
    r"""Simple tar archive extractor
    Args:
        file_path: path to the tar file to be extracted
        extract_folder: folder to which the files will be extracted
    """
    if extract_folder is None:
        extract_folder = os.path.dirname(file_path)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def download_untar(url, download_path, extract_path=None):
    r"""Download an archive from http link, extract it and then delete the archive"""
    file_name = url.split('/')[-1]
    if extract_path is None:
        extract_path = download_path
    tar_file_path = os.path.join(download_path, file_name)
    download(tar_file_path, url)
    sys.stdout.flush()
    print('Extracting {} archive into {}'.format(tar_file_path, extract_path))
    untar(tar_file_path, extract_path)
    os.remove(tar_file_path)
