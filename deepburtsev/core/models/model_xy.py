from .base_model import BaseModel


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
