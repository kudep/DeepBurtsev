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
