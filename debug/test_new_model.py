import numpy as np
import pymorphy2
import fasttext
import json
import os

from models.CNN.multiclass import KerasMulticlassModel as Model
morph = pymorphy2.MorphAnalyzer()

batch_size = 64
seq_len = 25
emb_dim = 100
X = np.random.rand(batch_size, seq_len, emb_dim)


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


def labels2onehot_one(labels, classes):
    n_classes = len(classes)
    eye = np.eye(n_classes)
    y = []
    for sample in labels:
        curr = eye[sample-1]
        y.append(curr)
    y = np.asarray(y)
    return y


y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = np.reshape(np.random.choice(y, batch_size*seq_len), [batch_size, seq_len])
print(Y)
print(Y.shape)

y_one = labels2onehot_one(Y[0], y)
print(y_one)
print(Y[0])



