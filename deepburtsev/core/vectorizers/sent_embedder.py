import tensorflow_hub as hub
import tensorflow as tf

from deepburtsev.core.transformers.transformers import BaseTransformer


class SentEmbedder(BaseTransformer):
    def __init__(self, config=None):
        if config is None:
            self.config = {'op_type': 'transformer',
                           'name': 'SentEmbedder',
                           'request_names': ['base'],
                           'new_names': ['base'],
                           'path': "https://tfhub.dev/google/universal-sentence-encoder/1"}
        else:
            need_names = ['path']
            for name in need_names:
                if name not in config.keys():
                    raise ValueError('Input config must contain {}.'.format(name))

            self.config = config

        super().__init__(self.config)
        self.model = hub.Module(self.config['path'])

    def _transform(self, dataset):
        print('[ SentEmbedder start working ... ]')
        request, report = dataset.main_names

        with tf.Session() as session:
            for name, new_name in zip(self.config['request_names'], self.config['new_names']):
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                message_embeddings = session.run(self.model(list(dataset.data[name][request])))

                print(message_embeddings.shape)

                dataset.data[new_name][request] = message_embeddings
                dataset.data[new_name][report] = list(dataset.data[name][report])
        return dataset
