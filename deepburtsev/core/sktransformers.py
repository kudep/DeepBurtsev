from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class Tfidf(TfidfVectorizer):
    def __init__(self, request_names='base', new_names='base', op_type='vectorizer', op_name='td-idf', **kwargs):
        super().__init__(**kwargs)

        if isinstance(request_names, list):
            self.request_names = request_names
        elif isinstance(request_names, str):
            self.request_names = [request_names]
        else:
            raise TypeError()

        if isinstance(new_names, list):
            self.new_names = new_names
        elif isinstance(new_names, str):
            self.new_names = [new_names]
        else:
            raise TypeError()

        self.op_type = op_type
        self.op_name = op_name

    def _fill_names(self, x_name, y_name):
        if x_name is not None:
            self.request_names = x_name
        if y_name is not None:
            self.new_names = y_name

        if len(self.new_names) != len(self.request_names):
            raise ValueError('The number of requested and new names must match.')

        return self

    def fit_transform(self, dataset, request_names=None, new_names=None):
        self._fill_names(request_names, new_names)
        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset[name]['x']
            Y = dataset[name]['y']
            if name == 'train':
                dataset[new_name]['x'] = super().fit_transform(X, Y)
            else:
                dataset[new_name]['x'] = super().transform(X)

        return dataset


class Count(CountVectorizer):
    def __init__(self, request_names='base', new_names='base', op_type='vectorizer', op_name='count-vectorizer',
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(request_names, list):
            self.request_names = request_names
        elif isinstance(request_names, str):
            self.request_names = [request_names]
        else:
            raise TypeError()

        if isinstance(new_names, list):
            self.new_names = new_names
        elif isinstance(new_names, str):
            self.new_names = [new_names]
        else:
            raise TypeError()

        self.op_type = op_type
        self.op_name = op_name

    def _fill_names(self, x_name, y_name):
        if x_name is not None:
            self.request_names = x_name
        if y_name is not None:
            self.new_names = y_name

        if len(self.new_names) != len(self.request_names):
            raise ValueError('The number of requested and new names must match.')

        return self

    def fit_transform(self, dataset, request_names=None, new_names=None):
        self._fill_names(request_names, new_names)
        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset[name]['x']
            Y = dataset[name]['y']

            if name == 'train':
                dataset[new_name]['x'] = super().fit_transform(X, Y)
            else:
                dataset[new_name]['x'] = super().transform(X)

        return dataset
