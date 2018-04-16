class BaseModel(object):
    def __init__(self, model, config):  # , fit_name=None, predict_names=None, new_names=None
        # config resist
        if not isinstance(config, dict):
            raise ValueError('Input config must be dict or None, but {} was found.'.format(type(config)))

        # keys = ['op_type', 'name', 'fit_names', 'predict_names', 'new_names', 'input_x_type', 'input_y_type',
        #         'output_x_type', 'output_y_type']

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

        self.config = config
        self.trained = False
        self.model_init = False
        self._validate_model(model)
        self.pure_model = model
        self.model = None

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
        if not self.model_init:
            self.model = self.pure_model(self.config)
            self.model_init = True
        else:
            # TODO it strange!
            if hasattr(self.model, 'reset'):
                self.save()
                # self.model.reset()
                self.model = self.pure_model(self.config)
            else:
                raise AttributeError('Model was already initialized. Add reset method in your model'
                                     'or create new pipeline')
        return self

    def fit(self, dataset, train_name=None):
        self._validate_names(dataset)
        self.init_model(dataset)

        if train_name is None:
            for name in self.fit_names:
                if hasattr(self.model, 'train'):
                    self.model.train(dataset, name)
                if hasattr(self.model, 'fit'):
                    self.model.fit(dataset, name)
        else:
            if hasattr(self.model, 'train'):
                self.model.train(dataset, train_name)
            if hasattr(self.model, 'fit'):
                self.model.fit(dataset, train_name)

        self.trained = True
        return self

    def predict(self, dataset, predict_name=None, new_name=None):
        self._validate_names(dataset)

        if predict_name is None and new_name is None:
            if not self.model_init:
                self.init_model(dataset)
            elif not self.trained:
                raise TypeError('Model is not trained yet.')
            else:
                for name, new_name in zip(self.request_names, self.new_names):
                    dataset.data[new_name] = self.model.predict(dataset, name)
        else:
            if not self.model_init:
                self.init_model(dataset)
            elif not self.trained:
                raise TypeError('Model is not trained yet.')
            else:
                dataset.data[new_name] = self.model.predict(dataset, predict_name)

        return dataset

    def predict_data(self, dataset, predict_name=None, new_name=None):
        self._validate_names(dataset)
        self.init_model(dataset)

        if not self.trained:
            raise TypeError('Model is not trained yet.')

        prediction = {}
        if predict_name is None and new_name is None:
            for name, new_name in zip(self.request_names, self.new_names):
                prediction[new_name] = self.model.predict(dataset, name)
        else:
            prediction[new_name] = self.model.predict(dataset, predict_name)

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
