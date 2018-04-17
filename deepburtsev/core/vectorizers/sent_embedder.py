import tensorflow_hub as hub
import tensorflow as tf

from deepburtsev.core.transformers.transformers import BaseTransformer


class SentEmbedder(BaseTransformer):
    def __init__(self, request_names=None, new_names=None, op_type='vectorizer', op_name='SentEmbedder',
                 model_path="https://tfhub.dev/google/universal-sentence-encoder/1"):
        super().__init__(request_names, new_names, op_type, op_name)
        self.model_path = model_path
        self.model = hub.Module(self.model_path)

    def _transform(self, dataset, request_names=None, new_names=None):
        print('[ SentEmbedder start working ... ]')

        if request_names is not None:
            self.worked_names = request_names
        if new_names is not None:
            self.new_names = new_names

        request, report = dataset.main_names

        with tf.Session() as session:
            for name, new_name in zip(self.config['request_names'], self.config['new_names']):
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                message_embeddings = session.run(self.model(list(dataset.data[name][request])))

                print(message_embeddings.shape)

                dataset.data[new_name][request] = message_embeddings
                dataset.data[new_name][report] = list(dataset.data[name][report])
        return dataset
