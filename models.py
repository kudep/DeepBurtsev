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
