import copy
from dataset import Dataset
from transformers import Speller, Tokenizer, Lemmatizer, FasttextVectorizer


class Pipeline(object):
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, dataset):

        for op in self.pipe:
            operation = op[1]
            if operation is not None:
                if operation.info['op_type'] == 'transformer':
                    dataset = operation.transform(dataset)
                elif operation.info['op_type'] == 'vectorizer':
                    if 'train' not in dataset.data.keys():
                        dataset.split()
                    operation.transform(dataset)
                elif operation.info['op_type'] == 'model':
                    operation.fit(dataset)
            else:
                pass

        print('[ Train End. ]')

        return self

    def predict(self, dataset):
        prediction = None

        for op in self.pipe:
            operation = op[1]
            if operation is not None:
                if operation.info['op_type'] == 'transformer':
                    dataset = operation.transform(dataset)
                elif operation.info['op_type'] == 'vectorizer':
                    if 'train' not in dataset.data.keys():
                        dataset.split()
                    operation.transform(dataset)
                elif operation.info['op_type'] == 'model':
                    prediction = operation.predict(dataset)
            else:
                pass

        print('[ Prediction End. ]')

        return prediction

    def run(self, fit_dataset, predict_dataset=None):
        self.fit(fit_dataset)
        if predict_dataset is None:
            prediction = self.predict(fit_dataset)
        else:
            prediction = self.predict(predict_dataset)

        print('[ End. ]')

        return prediction
