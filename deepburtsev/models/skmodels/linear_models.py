from sklearn.linear_model.logistic import LogisticRegression


class LinearRegression(LogisticRegression):
    def __init__(self, fit_name='train', pred_name='test', new_name='pred_test', op_type='model',
                 op_name='LogisticRegression', **kwargs):
        super().__init__(**kwargs)
        self.fit_name = fit_name
        self.pred_name = pred_name
        self.new_name = new_name
        self.op_type = op_type
        self.op_name = op_name

    def _fill_names(self, x_name, y_name, z_name):
        if x_name is not None:
            self.fit_name = x_name
        if y_name is not None:
            self.pred_name = y_name
        if z_name is not None:
            self.new_name = z_name

        return self

    def fit(self, dataset, fit_name=None, new_name=None):
        self._fill_names(fit_name, self.pred_name, new_name)

        X_fit = dataset[self.fit_name]['x']
        Y_fit = dataset[self.fit_name]['y']

        super().fit(X_fit, Y_fit, self.class_weight)

        return self

    def fit_transform(self, dataset, fit_name=None, pred_name=None, new_name=None):
        self._fill_names(fit_name, pred_name, new_name)

        X_fit = dataset[self.fit_name]['x']
        Y_fit = dataset[self.fit_name]['y']

        super().fit(X_fit, Y_fit, self.class_weight)

        X_pred = dataset[self.pred_name]['x']
        dataset[self.new_name] = {}
        dataset[self.new_name]['y_pred'] = super().predict(X_pred)
        dataset[self.new_name]['y_true'] = dataset[self.pred_name]['y']

        return dataset

    def predict(self, dataset, pred_name=None, new_name=None):
        self._fill_names(self.fit_name, pred_name, new_name)
        X_pred = dataset[self.pred_name]['x']
        dataset[self.new_name] = {}
        dataset[self.new_name]['y_pred'] = super().predict(X_pred)
        dataset[self.new_name]['y_true'] = dataset[self.pred_name]['y']

        return dataset
