import pandas as pd
import json
import re


def read_vkusvill_dataset(filepath, duplicates=False, clean=True):
    file = open(filepath, 'r', encoding='ISO-8859-1')
    data = pd.read_csv(file)

    old_names = data.keys()
    names = [n.encode('ISO-8859-1').decode('cp1251').encode('utf8') for n in old_names]
    names = [n.decode('utf-8') for n in names]

    new_data = dict()
    for old, new in zip(old_names, names):
        new_data[new] = list()
        for c in data[old]:
            try:
                s = c.encode('ISO-8859-1').decode('cp1251').encode('utf8')
                s = s.decode('utf-8')
                new_data[new].append(s)
            except AttributeError:
                new_data[new].append(c)

    new_data = pd.DataFrame(new_data, columns=['Описание', 'Категория жалобы'])
    new_data.rename(columns={'Описание': 'request', 'Категория жалобы': 'report'}, inplace=True)
    new_data = new_data.dropna()  # dell nan
    if not duplicates:
        new_data = new_data.drop_duplicates()  # dell duplicates

    # как отдельную ветвь можно использовать
    if clean:
        delete_bad_symbols = lambda x: " ".join(re.sub('[^а-яa-zё0-9]', ' ', x.lower()).split())
        new_data['request'] = new_data['request'].apply(delete_bad_symbols)

    new_data = new_data.reset_index()

    return new_data


def read_en_dataset(path, snips=False):

    if not snips:
        with open(path, 'r') as data:
            dataset = json.load(data)
            data.close()

        qestions = []
        classes_names = []
        for x in dataset['sentences']:
            qestions.append(x['text'])
            classes_names.append(x['intent'])

        categs = list(set(classes_names))
        classes = [categs.index(x) + 1 for x in classes_names]

        category_description = dict()
        for name in classes_names:
            category_description[name] = categs.index(name) + 1

        df = pd.DataFrame({'request': qestions, 'report': classes, 'names': classes_names})
    else:
        dataset = pd.read_csv(path)

        classes_names = list(dataset['intents'].unique())
        classes = [classes_names.index(x) + 1 for x in dataset['intents']]

        category_description = dict()
        for name in dataset['intents']:
            category_description[name] = classes_names.index(name) + 1

        df = pd.DataFrame({'request': dataset['text'], 'report': classes, 'names': dataset['intents']})
        del dataset

    return df, category_description


def read_snips_dataset(path):
    dataset = pd.read_csv(path)

    classes_names = list(dataset['intents'].unique())
    classes = [classes_names.index(x) + 1 for x in dataset['intents']]

    category_description = dict()
    for name in dataset['intents']:
        category_description[name] = classes_names.index(name) + 1

    df = pd.DataFrame({'request': dataset['text'], 'report': classes, 'names': dataset['intents']})
    del dataset

    return df, category_description


def read_sber_dataset(path):
    dataset = pd.read_csv(path)

    df = pd.DataFrame({'request': dataset['0'], 'report': dataset['1']})
    df = df.dropna()
    df = df.reset_index()
    del dataset

    return df