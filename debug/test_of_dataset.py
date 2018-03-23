#!/usr/bin/env python3
import pandas as pd
import re
from dataset import Dataset
from transformers import Speller, Tokenizer, Lemmatizer, FasttextVectorizer


def read_dataset(filepath, duplicates=False, clean=True):
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


# Reading full data
data_file = '../data/russian/data/vkusvill_all_categories.csv'
comment_name = "request"
cat_name = 'report'
train_data = read_dataset(data_file)
print(train_data.head())

# Building dataset
dataset = Dataset(data=train_data, seed=42)
# print(dataset.info())
# print(dataset.data['base'].head())
# print(dataset.data['train'])
# print(dataset.data['valid'])
# print(dataset.data['test'])
# print(dataset.data['base'])
# print('Len of Dataset: {}'.format(len(dataset.data['base'])))
#
# # splitting full dataset on train and valid in proportion 9:1
# dataset = dataset.split()
# print(dataset.info())
# # print(dataset.data['train'])
# # print(dataset.data['valid'])
# # print(dataset.data['test'])
# print('Len of sum fields: {}'.format(len(dataset.data['train']) + len(dataset.data['valid']) + len(dataset.data['test'])))
#
# dataset.simple_split(splitting_proportions=[0.5, 0.5], splitted_fields=['train_1', 'train_2'], field_to_split='train')
# print(dataset.info())
# print('Len of train_1: {}'.format(len(dataset.data['train_1'])))
# print('Len of train_2: {}'.format(len(dataset.data['train_2'])))
#
# dataset.merge_data(fields_to_merge=['train_1', 'train_2'], new_name='train')
# print(dataset.info())
# print('Len of train: {}'.format(len(dataset.data['train'])))
#
# dataset.merge_data(fields_to_merge=['train', 'valid', 'test'], new_name='base')
# print(dataset.info())
# print(len(dataset.data['base']))
# # print(dataset.data['base'])
#
classes = dataset.get_classes()
distribution = dataset.get_distribution()
print(classes, distribution)

dataset.simple_split([0.999, 0.001], 'base', ['base', 'test'], delete_parent=False)
print(dataset.info())
print(dataset.data['test'].head())

# Speller test:
# dataset = Speller().transform(dataset, name='test')
# print(dataset.data['test'].head())

# Tokenizer test:
dataset = Tokenizer().transform(dataset)
print(dataset.data['base'].head())

# dataset = Lemmatizer().transform(dataset)
# print(dataset.data['base'].head())

dataset.split()
print(dataset.info())

dataset = FasttextVectorizer().transform(dataset, name='train')
print(dataset.data['train'].head(2))


