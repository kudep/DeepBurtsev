from transformer import BaseTransformer


class skwrapper(BaseTransformer):
    def __init__(self, t, config=None):
        super().__init__(config)

        self.transformer = t()
        if not ((hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
            raise TypeError("Methods fit, fit_transform, transform are not implemented in class {} "
                            "Sklearn transformers and estimators shoud implement fit and transform.".format(t))

        self.trained = False
        params = self.transformer.get_params()
        for key in params.keys():
            if key in self.config.keys():
                params[key] = self.config[key]

        self.transformer.set_params(**params)


class sktransformer(skwrapper):
    def __init__(self, t, config=None):
        super().__init__(t, config)

    def _transform(self, dataset):
        request, report = dataset.main_names
        if hasattr(self.transformer, 'fit_transform') and not self.trained:
            if 'base' not in dataset.data.keys():
                dataset.merge_data(fields_to_merge=self.request_names, delete_parent=False, new_name='base')
                X = dataset.data['base'][request]
                y = dataset.data['base'][report]
                # fit
                self.transformer.fit(X, y)
                self.trained = True

                # delete 'base' from dataset
                dataset.del_data(['base'])
            else:
                X = dataset.data['base'][request]
                y = dataset.data['base'][report]
                # fit
                self.transformer.fit(X, y)
                self.trained = True

            # transform all fields
            for name, new_name in zip(self.request_names, self.new_names):
                X = dataset.data[name][request]
                y = dataset.data[name][report]
                dataset.data[new_name] = {request: self.transformer.transform(X),
                                          report: y}

        else:
            for name, new_name in zip(self.request_names, self.new_names):
                X = dataset.data[name][request]
                y = dataset.data[name][report]
                dataset.data[new_name] = {request: self.transformer.transform(X),
                                          report: y}

        return dataset


class skmodel(skwrapper):
    def __init__(self, t, config=None):
        super().__init__(t, config)

    def fit(self, dataset):
        request, report = dataset.main_names

        if 'train_vec' in dataset.data.keys():
            name = 'train_vec'
        else:
            if 'train' in dataset.data.keys():
                name = 'train'
            else:
                raise KeyError('Dataset must contain "train_vec" or "train" fields.')

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

        if (request_names is not None) and (new_names):
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

        if (request_names is not None) and (new_names):
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


class BaseModel(object):
    def __init__(self, model, config):  # , fit_name=None, predict_names=None, new_names=None
        # config resist
        if not isinstance(config, dict):
            raise ValueError('Input config must be dict or None, but {} was found.'.format(type(config)))

        keys = ['op_type', 'name', 'fit_names', 'predict_names', 'new_names', 'input_x_type', 'input_y_type',
                'output_x_type', 'output_y_type']

        self.info = dict()

        for x in keys:
            if x not in config.keys():
                raise ValueError('Input config must contain {} key.'.format(x))

            elif x == 'fit_names' or x == 'predict_names' or x == 'new_names':
                if not isinstance(config[x], list):
                    raise ValueError('Parameters fit_names, predict_names and new_names in config must be a list,'
                                     ' but {} "{}" was found.'.format(type(config[x]), config[x]))
            self.info[x] = config[x]

        self.config = config
        self.trained = False
        self._validate_model(model)
        self.model = model

        # named spaces
        self.new_names = config['new_names']
        self.fit_names = config['fit_names']
        self.request_names = config['predict_names']

    def _validate_names(self, dataset):
        if self.fit_names is not None:
            for name in self.fit_names:
                if name not in dataset.data.keys():
                    raise KeyError('Key {} not found in dataset.'.format(name))
        else:
            raise KeyError('Parameter fit_names in config can not be None.')

        if self.request_names is not None:
            for name in self.request_names:
                if name not in dataset.data.keys():
                    raise KeyError('Key {} not found in dataset.'.format(name))

        return self

    def _validate_model(self, model):
        # need_atr = ['fit', 'predict', 'fit_predict', 'save', 'restore']
        # for atr in need_atr:
        #     if not hasattr(model, atr):
        #         raise AttributeError("Model don't supported {} method.".format(atr))

        if not (hasattr(model, 'fit') or hasattr(model, 'train')):
            raise AttributeError("Model don't supported fit or train methods method.")
        elif not (hasattr(model, 'restore') or hasattr(model, 'load')):
            raise AttributeError("Model don't supported restore or load methods method.")
        elif not hasattr(model, 'save'):
            raise AttributeError("Model don't supported save methods method.")
        elif not hasattr(model, 'predict'):
            raise AttributeError("Model don't supported predict or load methods method.")

        return self

    def init_model(self, dataset):
        self.model = self.model(self.config)
        return self

    def fit(self, dataset):
        self._validate_names(dataset)
        self.init_model(dataset)

        for name in self.fit_names:
            if hasattr(self.model, 'train'):
                self.model.train(dataset, name)
            if hasattr(self.model, 'fit'):
                self.model.fit(dataset, name)

        return self

    def predict(self, dataset):
        self._validate_names(dataset)
        self.init_model(dataset)
        if not self.trained:
            raise TypeError('Model is not trained yet.')
        for name, new_name in zip(self.request_names, self.new_names):
            dataset.data[new_name] = self.model.predict(dataset, name)

        return dataset

    def predict_data(self, dataset):
        self._validate_names(dataset)
        self.init_model(dataset)

        if not self.trained:
            raise TypeError('Model is not trained yet.')

        prediction = {}
        for name, new_name in zip(self.request_names, self.new_names):
            prediction[new_name] = self.model.predict(dataset, name)

        return prediction

    def fit_predict(self, dataset):
        self.fit(dataset)
        dataset = self.predict(dataset)
        return dataset

    def fit_predict_data(self, dataset):
        self.fit(dataset)
        prediction = self.predict_data(dataset)
        return prediction

    def get_params(self):
        return self.config

    def set_params(self, params):
        self.config = params
        return self

    def save(self, path=None):
        if path is not None:
            self.model.save(path)
        else:
            self.model.save()
        return self

    def restore(self, path=None):
        if path is not None:
            if isinstance(path, str):
                self.model.restore(path)
                self.trained = True
            else:
                raise TypeError('Restore path must be str, but {} was found.'.format(type(path)))
        else:
            self.model.restore()
            self.trained = True

        return self


class ModelXY(BaseModel):
    def __init__(self, model, config):
        super().__init__(model, config)

    def fit(self, dataset):
        self._validate_names(dataset)
        request, report = dataset.main_names

        for name in self.fit_names:
            X = dataset.data[name][request]
            Y = dataset.data[name][report]
            self.model.fit(X, Y)

        return self

    def predict(self, dataset):
        self._validate_names(dataset)
        request, report = dataset.main_names

        if not self.trained:
            raise TypeError('Model is not trained yet.')

        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset.data[name][request]
            dataset.data[new_name] = self.model.predict(X)

        return dataset

    def predict_data(self, dataset):
        self._validate_names(dataset)
        request, report = dataset.main_names

        if not self.trained:
            raise TypeError('Model is not trained yet.')

        prediction = {}
        for name, new_name in zip(self.request_names, self.new_names):
            X = dataset.data[name][request]
            prediction[new_name] = self.model.predict(X)

        return prediction

    def fit_predict(self, dataset):
        self.fit(dataset)
        dataset = self.predict(dataset)
        return dataset

    def fit_predict_data(self, dataset):
        self.fit(dataset)
        prediction = self.predict_data(dataset)
        return prediction
