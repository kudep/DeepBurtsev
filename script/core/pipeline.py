class BasePipeline(object):
    def __init__(self, pipe, mode='train', output=None):
        self.pipe = pipe
        self.mode = mode
        self.output = output

        if self.mode == 'train' and self.output == 'predictions':
            raise AttributeError("Pipeline can't returning predictions of model in train mode.")
        elif self.mode == 'infer' and self.output is None:
            raise AttributeError("Pipeline must returning predictions or dataset object in infer mode.")

        self.pipeline_config = {'pipeline': {'mode': self.mode, 'output': self.output}}

    # TODO write the super power resist
    # def _validate_pipeline(self):
    #     for i, op in enumerate(self.pipe):

    def step(self, i, op, dataset, last_op=False):
        if len(op) == 1:
            operation = op[0]()
            op_type = operation.info['op_type']
            self.config_constructor(operation.config, i)
        elif len(op) == 2:
            if op[1] is None:
                operation = op[0]()
                op_type = operation.info['op_type']
                self.config_constructor(operation.config, i)
            else:
                if not isinstance(op[1], dict):
                    raise AttributeError('Config of operation {0} must be a dict,'
                                         ' but {1} was found.'.format(op, type(op)))
                op_type = op[1]['op_type']
                operation = op[0](config=op[1])
                self.config_constructor(op[1], i)
        else:
            raise AttributeError('Operation in pipeline input list must be tuple like (operation, config), '
                                 'but {0} was found, with length={1}.'.format(op, len(op)))

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
                elif self.output == 'predictions':
                    if last_op:
                        dataset_ = operation.fit_predict_data(dataset)
                    else:
                        dataset_ = operation.fit_predict(dataset)
                else:
                    raise AttributeError('Pipeline must returning predictions or dataset object in infer mode,'
                                         'but {} was found.'.format(self.output))

                return dataset_

            elif self.mode == 'train':
                if last_op:
                    operation.fit(dataset)
                    return None
                else:
                    dataset_ = operation.fit_predict(dataset)
                    return dataset_

            else:
                raise AttributeError('Pipeline can be only in infer or train mode,'
                                     ' but {} was found.'.format(self.mode))
        else:
            raise AttributeError('It can not happened.')

    def config_constructor(self, conf, num):
        conf['num_op'] = num
        key = conf['op_type'] + '_' + conf['name']
        self.pipeline_config[key] = conf
        return self

    def get_last_model(self):
        pass

    def get_models(self):
        pass

    def run(self, dataset):
        dataset_i = dataset
        for i, op in enumerate(self.pipe):
            if i == len(self.pipe) - 1:
                dataset_i = self.step(i, op, dataset_i, last_op=True)
            else:
                dataset_i = self.step(i, op, dataset_i)
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


pipeline_config = {'mode': 'train',  # [train, infer]
                   'return': 'None'}  # [None, dataset, predictions]
