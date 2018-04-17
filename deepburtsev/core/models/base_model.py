from ..transformers.transformers import BaseClass


class BaseModel(BaseClass):
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

        # TODO init check
        if not (hasattr(model, 'fit') or hasattr(model, 'train')):
            raise AttributeError("Model don't supported fit or train methods method.")
        elif not (hasattr(model, 'restore') or hasattr(model, 'load')):
            raise AttributeError("Model don't supported restore or load methods method.")
        elif not hasattr(model, 'save'):
            raise AttributeError("Model don't supported save methods method.")
        elif not hasattr(model, 'predict'):
            raise AttributeError("Model don't supported predict or load methods method.")

        return self
