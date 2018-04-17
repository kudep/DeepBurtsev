from .base_model import BaseModel


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
