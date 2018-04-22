from deepburtsev.core.transformers import BaseTransformer


class skwrapper(BaseTransformer):
    def __init__(self, t, request_names=None, new_names=None, op_type=None, op_name=None, **kwargs):
        super().__init__(request_names, new_names, op_type, op_name)
        self.trained = False

        self.transformer = t(**kwargs)
        if not ((hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
            raise TypeError("Methods fit, fit_transform, transform are not implemented in class {} "
                            "Sklearn transformers and estimators shoud implement fit and transform.".format(t))


class skmodel(skwrapper):
    def __init__(self, t, config=None):

        self.transformer = t()
        if not ((hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
            raise TypeError("Methods fit, fit_transform, transform are not implemented in class {} "
                            "Sklearn transformers and estimators shoud implement fit and transform.".format(t))
        # config resist
        if not isinstance(config, dict):
            raise ValueError('Input config must be dict or None, but {} was found.'.format(type(config)))

        keys = ['op_type', 'name', 'fit_names', 'predict_names', 'new_names']

        self.info = dict()

        for x in keys:
            if x not in config.keys():
                raise ValueError('Input config must contain {} key.'.format(x))
            self.info[x] = config[x]

        for x in keys:
            if x == 'fit_names' or x == 'predict_names' or x == 'new_names':
                if not isinstance(config[x], list):
                    raise ValueError('Parameters fit_names, predict_names and new_names in config must be a list,'
                                     ' but {} "{}" was found.'.format(type(config[x]), config[x]))
        # named spaces
        self.new_names = config['new_names']
        self.fit_names = config['fit_names']
        self.request_names = config['predict_names']

        self.config = config
        self.trained = False
        params = self.transformer.get_params()
        for key in params.keys():
            if key in self.config.keys():
                params[key] = self.config[key]

        self.transformer.set_params(**params)

    def fit(self, dataset):
        request, report = dataset.main_names

        for name in self.fit_names:
            X = dataset.data[name][request]
            y = dataset.data[name][report]

            if hasattr(self.transformer, 'fit') and not hasattr(self.transformer, 'fit_tranform'):
                self.transformer.fit(X, y)
                self.trained = True

        return self

    def predict(self, dataset, request_names=None, new_names=None):

        if not hasattr(self.transformer, 'predict'):
            raise TypeError("Methods predict, is not implemented in class {} "
                            " '%s' (type %s) doesn't" % (self.transformer, type(self.transformer)))

        request, report = dataset.main_names

        if not self.trained:
            raise AttributeError('Sklearn model is not trained yet.')

        if (request_names is not None) and (new_names is not None):
            if not isinstance(request_names, list) or not isinstance(new_names, list):
                raise AttributeError('Parameters request_names and new_names must be a list,'
                                     'but {} was found.'.format(type(request_names), type(new_names)))
            self.request_names = request_names
            self.new_names = new_names

        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset.data[name][request]
            dataset.data[new_name] = self.transformer.predict(X)

        return dataset

    def fit_predict(self, dataset, request_names=None, new_names=None):
        self.fit(dataset)
        dataset = self.predict(dataset, request_names, new_names)
        return dataset

    def predict_data(self, dataset, request_names=None, new_names=None):

        if not hasattr(self.transformer, 'predict'):
            raise TypeError("Methods predict, is not implemented in class {} "
                            " '%s' (type %s) doesn't" % (self.transformer, type(self.transformer)))

        request, report = dataset.main_names

        if not self.trained:
            raise AttributeError('Sklearn model is not trained yet.')

        if (request_names is not None) and (new_names is not None):
            if not isinstance(request_names, list) or not isinstance(new_names, list):
                raise AttributeError('Parameters request_names and new_names must be a list,'
                                     'but {} was found.'.format(type(request_names), type(new_names)))
            self.request_names = request_names
            self.new_names = new_names

        res = []
        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset.data[name][request]
            res.append(self.transformer.predict(X))

        return res

    def fit_predict_data(self, dataset, request_names=None, new_names=None):
        self.fit(dataset)
        res = self.predict_data(dataset, request_names, new_names)
        return res
