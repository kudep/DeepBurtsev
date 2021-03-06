from deepburtsev.core.transformers import BaseClass


class BaseModel(BaseClass):
    def __init__(self, fit_name=None, predict_names=None, new_names=None, op_type='model',
                 op_name='base_model'):
        self.op_type = op_type
        self.op_name = op_name

        # named spaces
        self.new_names = new_names
        self.fit_name = fit_name
        self.request_names = predict_names

        self.trained = False
        self.model_init = False
        self.model = None

    def _validate_names(self, dataset):
        if self.fit_name is not None:
            for name in self.fit_name:
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
        # TODO init check
        if not (hasattr(model, 'fit') or hasattr(model, 'train')):
            raise AttributeError("Model don't supported fit or train methods method.")
        elif not (hasattr(model, 'restore') or hasattr(model, 'load')):
            raise AttributeError("Model don't supported restore or load methods method.")
        elif not hasattr(model, 'save'):
            raise AttributeError("Model don't supported save methods method.")
        elif not hasattr(model, 'predict'):
            raise AttributeError("Model don't supported predict or load methods method.")

    def _fill_names(self, fit, pred, new):
        if fit is not None:
            if isinstance(fit, str):
                self.fit_name = fit
            else:
                raise ValueError('Fit name must be str, but {} was found.'.format(type(fit)))

        if pred is not None:
            if isinstance(pred, list):
                self.request_names = pred
            elif isinstance(pred, str):
                self.request_names = [pred]
            else:
                raise ValueError('Prediction names must be a string or a list of strings.')

        if new is not None:
            if isinstance(new, list):
                self.new_names = new
            elif isinstance(pred, str):
                self.new_names = [new]
            else:
                raise ValueError('New names must be a string or a list of strings.')

        if isinstance(self.new_names, str):
            self.new_names = [self.new_names]
        if isinstance(self.request_names, str):
            self.request_names = [self.request_names]

        if len(self.new_names) != len(self.request_names):
            raise ValueError('Lists with predicted names and new names must have equal length.')
        return self


class Model(BaseModel):
    def __init__(self, model, fit_name=None, predict_names=None, new_names=None, op_type='model',
                 op_name='model'):
        super().__init__(fit_name, predict_names, new_names, op_type,
                         op_name)

        self.model_ = model

    def init_model(self):
        if not self.model_init:
            self.model = self.model_(**self.config)
            self.model_init = True
        else:
            # TODO it strange!
            if hasattr(self.model, 'reset'):
                self.save()
                # self.model.reset()
                self.model = self.model_(self.config)
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


class KerasModel(BaseModel):

    def init_model(self, dataset):
        if not self.model_init:
            # compilation
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics_funcs,
                               loss_weights=None,
                               sample_weight_mode=None,
                               # weighted_metrics=weighted_metrics,
                               # target_tensors=target_tensors
                               )

            self.metrics_names = self.model.metrics_names
            self.metrics_values = len(self.metrics_names) * [0.]

            self.model_init = True
        else:
            raise AttributeError('Model was already initialized. Add reset method in your model'
                                 'or create new pipeline')
        return self

    # TODO rewrite as hole train function
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
