# only for python 3.3+
from inspect import signature
from collections import defaultdict


class BaseModel(object):
    def __init__(self, fit_name=None, predict_names=None, new_names=None, op_type='model',
                 op_name='base_model'):
        self.op_type = op_type
        self.op_name = op_name

        # named spaces
        self.new_names = new_names
        self.fit_names = fit_name
        self.request_names = predict_names

        self.trained = False
        self.model_init = False
        # self._validate_model(model)
        # self.model_ = model
        self.model = None

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

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
            self.model = self.model_(self.config)
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
