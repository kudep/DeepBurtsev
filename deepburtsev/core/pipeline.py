class Pipeline(object):
    def __init__(self, pipe):
        self.pipe = pipe
        self.models = []

    # TODO modify this inspection
    def _check_pipe(self, pipe):
        for i, element in enumerate(pipe):
            if isinstance(element, tuple):
                if not isinstance(element[1], dict):
                    raise ValueError('In {0} element of input list, was found {1},'
                                     'but need dict of params.'.format(i+1, type(element[1])))
        return self

    # def step_from(self,)

    def step(self, element, dataset):
        if isinstance(element, tuple):
            op = element[0](**element[1])
        else:
            op = element()

        dataset_ = op.fit_predict(dataset)
        return dataset_

    def run(self, dataset):
        dataset_i = dataset
        for i, element in enumerate(self.pipe):
            try:
                dataset_i = self.step(element, dataset_i)
            except:
                print('Operation with number {0};'.format(i + 1))
                raise
        out = dataset_i
        return out

    def get_last_model(self):
        return self.models[-1]

    def get_models(self):
        return self.models

    def fit(self, dataset):
        out = self.run(dataset)
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
