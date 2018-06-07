class Pipeline(object):
    def __init__(self, pipe):
        self.pipe = pipe
        self._check_pipe(self.pipe)
        self.length = len(pipe)
        self.models = []

    def get_op_config(self, i):
        element = self.pipe[i]
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                try:
                    op = element[0](**element[1])
                except TypeError:
                    print(element[1])
                    raise
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        conf = op.get_params()
        return conf

    # TODO modify this inspection
    def _check_pipe(self, pipe):
        for i, element in enumerate(pipe):
            if isinstance(element, tuple):
                if not isinstance(element[1], dict):
                    raise ValueError('In {0} element of input list, was found {1},'
                                     'but need dict of params.'.format(i+1, type(element[1])))
                if not hasattr(element[0], "fit_transform"):
                    raise TypeError("All intermediate steps should be "
                                    "transformers and implement fit and transform."
                                    " '%s' (type %s) doesn't" % (element[0], type(element[0])))
            else:
                if not hasattr(element, "fit_transform"):
                    raise TypeError("All intermediate steps should be "
                                    "transformers and implement fit and transform."
                                    " '%s' (type %s) doesn't" % (element, type(element)))

        return self

    def start_from(self, n, dataset):
        dataset_i = dataset
        for i, element in enumerate(self.pipe[n-1:]):
            try:
                dataset_i = self.step_(element, dataset_i)
            except:
                print('Operation with number: {0}; and name: {1}'.format(i + 1, element.op_name))
                raise
        out = dataset_i
        return out

    def step_(self, element, dataset):
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                op = element[0](**element[1])
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        dataset_ = op.fit_transform(dataset)

        # collecting models
        if op.op_type == 'model':
            self.models.append(op)

        return dataset_

    def step(self, i, dataset):
        element = self.pipe[i]
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                op = element[0](**element[1])
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        dataset_ = op.fit_transform(dataset)

        # collecting models
        if op.op_type == 'model':
            self.models.append(op)

        return dataset_

    def run(self, dataset):
        dataset_i = dataset
        for i, element in enumerate(self.pipe):
            try:
                dataset_i = self.step_(element, dataset_i)
            except:
                print('Operation with number: {0}; and name: {1}'.format(i + 1, element.op_name))
                raise
        out = dataset_i
        return out

    def get_last_model(self):
        return self.models[-1]

    def get_models(self):
        return self.models

    def fit(self, dataset):
        out = self.run(dataset)
        del out
        print('[ Train End. ]')

        return self

    def predict(self, dataset):
        prediction = self.run(dataset)
        print('[ Prediction End. ]')

        return prediction

    def predict_data(self, dataset):
        prediction = self.run(dataset)
        print('[ Prediction End. ]')

        return prediction


class MemPipeline(object):
    def __init__(self, pipe):
        self.pipe = pipe
        self.length = len(pipe)
        self._check_pipe(self.pipe)
        self.mem = dict()
        self.models = []

    def get_op_config(self, i):
        element = self.pipe[i]
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                try:
                    op = element[0](**element[1])
                except TypeError:
                    print(element[1])
                    raise
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        conf = op.get_params()
        return conf

    # TODO modify this inspection
    def _check_pipe(self, pipe):
        for i, element in enumerate(pipe):
            if isinstance(element, tuple):
                if not isinstance(element[1], dict):
                    raise ValueError('In {0} element of input list, was found {1},'
                                     'but need dict of params.'.format(i+1, type(element[1])))
                if not hasattr(element[0], "fit_transform"):
                    raise TypeError("All intermediate steps should be "
                                    "transformers and implement fit and transform."
                                    " '%s' (type %s) doesn't" % (element[0], type(element[0])))
            else:
                if not hasattr(element, "fit_transform"):
                    raise TypeError("All intermediate steps should be "
                                    "transformers and implement fit and transform."
                                    " '%s' (type %s) doesn't" % (element, type(element)))

        return self

    def start_from(self, n, dataset):
        dataset_i = dataset
        for i, element in enumerate(self.pipe[n-1:]):
            try:
                dataset_i = self.step_(element, dataset_i)
            except:
                print('Operation with number: {0}; and name: {1}'.format(i + 1, element.op_name))
                raise
        out = dataset_i
        return out

    def step_(self, element, dataset):
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                op = element[0](**element[1])
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        dataset_ = op.fit_transform(dataset)

        # collecting models
        if op.op_type == 'model':
            self.models.append(op)

        return dataset_

    def step(self, i, dataset):
        element = self.pipe[i]
        if isinstance(element, tuple):
            if not hasattr(element[0], '__call__'):
                op = element[0].set_params(**element[1])
            else:
                op = element[0](**element[1])
        else:
            if not hasattr(element, '__call__'):
                op = element
            else:
                op = element()

        dataset_ = op.fit_transform(dataset)

        # collecting models
        if op.op_type == 'model':
            self.models.append(op)

        return dataset_

    def run(self, dataset):
        dataset_i = dataset
        for i, element in enumerate(self.pipe):
            try:
                dataset_i = self.step_(element, dataset_i)
            except:
                print('Operation with number: {0}; and name: {1}'.format(i + 1, element.op_name))
                raise
        out = dataset_i
        return out

    def get_last_model(self):
        return self.models[-1]

    def get_models(self):
        return self.models

    def fit(self, dataset):
        out = self.run(dataset)
        del out
        print('[ Train End. ]')

        return self

    def predict(self, dataset):
        prediction = self.run(dataset)
        print('[ Prediction End. ]')

        return prediction

    def predict_data(self, dataset):
        prediction = self.run(dataset)
        print('[ Prediction End. ]')

        return prediction
