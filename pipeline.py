import copy
from dataset import Dataset
from transformer import Speller, Tokenizer, Lemmatizer, FasttextVectorizer


class Pipeline(object):
    def __init__(self, pipe, config=None):
        self.pipe = []
        self.config = None

        if config is not None:
            assert len(pipe) == len(config), ('List of operations and configurations has different length:'
                                              'pipe={0}, config={1}'.format(len(pipe), len(config)))
            for i, x, y in enumerate(zip(pipe, config)):
                assert len(y) == 2, ('Config of operation must be tuple of len two:'
                                     'len of {0} element in configs={1};'.format(i, len(y[i])))
                assert x[0] == y[0], ('Names of operations and configurations must be the same:'
                                      'pipe_{0}={1}, config_{0}={2};'.format(i, x[0], y[0]))

            for i, x, y in enumerate(zip(pipe, config)):
                self.pipe.append(x[1].__init__(y[1]))

            self.config = config
        else:
            for op in pipe:
                self.pipe.append(op[1].__init__())

        self._validate_steps()

    # def _validate_names(self, names):
    #     if len(set(names)) != len(names):
    #         raise ValueError('Names provided are not unique: '
    #                          '{0!r}'.format(list(names)))
    #     invalid_names = set(names).intersection(self.get_params(deep=False))
    #     if invalid_names:
    #         raise ValueError('Estimator names conflict with constructor '
    #                          'arguments: {0!r}'.format(sorted(invalid_names)))
    #     invalid_names = [name for name in names if '__' in name]
    #     if invalid_names:
    #         raise ValueError('Estimator names must not contain __: got '
    #                          '{0!r}'.format(invalid_names))

    def _validate_steps(self):
        # validate names
        # self._validate_names(names)

        # validate models and transformers
        transformers = []
        models = []
        for op in self.pipe:
            if op.info['type'] == 'transformer':
                transformers.append(op)
            elif op.info['type'] == 'model':
                models.append(op)
            else:
                raise TypeError("All operations should be transformers or models and implement fit and transform,"
                                "but {0} operation has type: {1}".format(op, op.info['type']))

        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        for m in models:
            if m is not None and not (hasattr(m, "fit") or hasattr(m, "predict")):
                raise TypeError("Models should implement fit or predict. "
                                "'%s' (type %s) doesn't"
                                % (m, type(m)))

    def fit(self, dataset, **fit_params):
        for op in self.pipe:
            if op is not None:
                if op.info['type'] == 'transformer':
                    dataset = op.transform(dataset, name='base')
                elif op.info['type'] == 'model':
                    op.init(dataset)
                    op.fit()
            else:
                pass

        return self

    def predict(self, dataset):
        prediction = None

        for op in self.pipe:
            if op is not None:
                if op.info['type'] == 'transformer':
                    dataset = op.transform(dataset, name='base')
                elif op.info['type'] == 'model':
                    op.init(dataset)
                    prediction = op.predict(dataset)
            else:
                pass

        return prediction


def _fit_one_transformer(transformer, X, y):
    return transformer.fit(X, y)


def _transform_one(transformer, weight, X):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer, weight, X, y,
                       **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer
