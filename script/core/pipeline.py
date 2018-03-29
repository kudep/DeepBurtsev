from collections import OrderedDict


class Pipeline(object):
    def __init__(self, pipe, mode='train', output=None):
        self.pipe = pipe
        self.mode = mode
        self.output = output
        self.models = []

        if self.mode == 'train' and self.output == 'predictions':
            raise AttributeError("Pipeline can't returning predictions of model in train mode.")
        elif self.mode == 'infer' and self.output is None:
            raise AttributeError("Pipeline must returning predictions or dataset object in infer mode.")

        self.pipeline_config = OrderedDict(pipeline={'mode': self.mode, 'output': self.output})

    # TODO write the super power resist
    # def _validate_pipeline(self):
    #     for i, op in enumerate(self.pipe):

    def step(self, i, op, dataset, last_op=False):
        if len(op) == 1:
            operation = op[0]
            try:
                op_type = operation.info['op_type']
                self.config_constructor(operation.config, i)
            except AttributeError:
                operation = op[0]()
                op_type = operation.info['op_type']
                self.config_constructor(operation.config, i)
        elif len(op) == 2:
            if op[1] is None:
                operation = op[0]
                try:
                    op_type = operation.info['op_type']
                    self.config_constructor(operation.config, i)
                except AttributeError:
                    operation = op[0]()
                    op_type = operation.info['op_type']
                    self.config_constructor(operation.config, i)
            else:
                if not isinstance(op[1], dict):
                    raise AttributeError('Config of operation {0} must be a dict,'
                                         ' but {1} was found.'.format(op, type(op)))
                op_type = op[1]['op_type']
                try:
                    operation = op[0](config=op[1])
                    self.config_constructor(op[1], i)
                except TypeError:
                    operation = op[0].set_params(op[1])
                    self.config_constructor(op[1], i)
        else:
            raise AttributeError('Operation in pipeline input list must be tuple like (operation, config), '
                                 'but {0} was found, with length={1}.'.format(op, len(op)))

        dataset.add_config(operation.config)

        # logic of methods calls
        if op_type == 'transformer':
            dataset_ = operation.transform(dataset)
            return dataset_
        elif op_type == 'vectorizer':
            # TODO i don't like it
            if 'train' not in dataset.data.keys():
                dataset = dataset.split()
            dataset_ = operation.transform(dataset)
            return dataset_
        elif op_type == 'model':
            if self.mode == 'infer':
                if self.output == 'dataset':
                    dataset_ = operation.fit_predict(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                elif self.output == 'predictions':
                    if last_op:
                        dataset_ = operation.fit_predict_data(dataset)
                        # add model
                        self.models.append((operation.info['name'], operation))
                    else:
                        dataset_ = operation.fit_predict(dataset)
                        # add model
                        self.models.append((operation.info['name'], operation))
                else:
                    raise AttributeError('Pipeline must returning predictions or dataset object in infer mode,'
                                         'but {} was found.'.format(self.output))

                return dataset_

            elif self.mode == 'train':
                if last_op:
                    operation.fit(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                    return None
                else:
                    dataset_ = operation.fit_predict(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                    return dataset_

            else:
                raise AttributeError('Pipeline can be only in infer or train mode,'
                                     ' but {} was found.'.format(self.mode))
        else:
            raise AttributeError('It can not happened.')

    def config_constructor(self, conf, num):
        conf['num_op'] = num

        # op_type = conf.pop('op_type')
        # name = conf.pop('name')

        key = conf['name'] + '_' + conf['op_type']
        self.pipeline_config[key] = conf
        return self

    def get_last_model(self):
        return self.models[-1]

    def get_models(self):
        return self.models

    def run(self, dataset):
        dataset_i = dataset
        for i, op in enumerate(self.pipe):
            try:
                if i == len(self.pipe) - 1:
                    dataset_i = self.step(i, op, dataset_i, last_op=True)
                else:
                    dataset_i = self.step(i, op, dataset_i)
            except:
                print('Operation with number {0};'.format(i + 1))
                raise
        out = dataset_i
        return out

    def fit(self, dataset):
        self.mode = 'train'
        self.output = None
        out = self.run(dataset)
        print('[ Train End. ]')

        return self

    def predict(self, dataset):
        self.mode = 'infer'
        self.output = 'dataset'
        prediction = self.run(dataset)

        print('[ Prediction End. ]')

        return prediction

    def predict_data(self, dataset):
        self.mode = 'infer'
        self.output = 'predictions'
        prediction = self.run(dataset)

        print('[ Prediction End. ]')

        return prediction


class PrepPipeline(Pipeline):
    def __init__(self, pipe, mode='train', output=None):
        super().__init__(pipe, mode, output)

    def step(self, i, op, dataset, last_op=False):
        if len(op) == 1:
            operation = op[0]
            try:
                op_type = operation.info['op_type']
                self.config_constructor(operation.config, i)
            except AttributeError:
                operation = op[0]()
                op_type = operation.info['op_type']
                self.config_constructor(operation.config, i)
        elif len(op) == 2:
            if op[1] is None:
                operation = op[0]
                try:
                    op_type = operation.info['op_type']
                    self.config_constructor(operation.config, i)
                except AttributeError:
                    operation = op[0]()
                    op_type = operation.info['op_type']
                    self.config_constructor(operation.config, i)
            else:
                if not isinstance(op[1], dict):
                    raise AttributeError('Config of operation {0} must be a dict,'
                                         ' but {1} was found.'.format(op, type(op)))
                op_type = op[1]['op_type']
                try:
                    operation = op[0](config=op[1])
                    self.config_constructor(op[1], i)
                except TypeError:
                    operation = op[0].set_params(op[1])
                    self.config_constructor(op[1], i)
        else:
            raise AttributeError('Operation in pipeline input list must be tuple like (operation, config), '
                                 'but {0} was found, with length={1}.'.format(op, len(op)))

        # logic of methods calls
        if op_type == 'transformer':

            have = dataset.test_config(operation.config)
            if not have:
                dataset_ = operation.transform(dataset)
                return dataset_
            else:
                return dataset

        elif op_type == 'vectorizer':
            # TODO i don't like it
            if 'train' not in dataset.data.keys():
                dataset = dataset.split()

            have = dataset.test_config(operation.config)
            if not have:
                dataset_ = operation.transform(dataset)
                return dataset_
            else:
                return dataset

        elif op_type == 'model':
            if self.mode == 'infer':
                if self.output == 'dataset':
                    dataset_ = operation.fit_predict(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                elif self.output == 'predictions':
                    if last_op:
                        dataset_ = operation.fit_predict_data(dataset)
                        # add model
                        self.models.append((operation.info['name'], operation))
                    else:
                        dataset_ = operation.fit_predict(dataset)
                        # add model
                        self.models.append((operation.info['name'], operation))
                else:
                    raise AttributeError('Pipeline must returning predictions or dataset object in infer mode,'
                                         'but {} was found.'.format(self.output))

                return dataset_

            elif self.mode == 'train':
                if last_op:
                    dataset_ = operation.fit(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                    return None
                else:
                    dataset_ = operation.fit_predict(dataset)
                    # add model
                    self.models.append((operation.info['name'], operation))
                    return dataset_

            else:
                raise AttributeError('Pipeline can be only in infer or train mode,'
                                     ' but {} was found.'.format(self.mode))
        else:
            raise AttributeError('It can not happened.')
