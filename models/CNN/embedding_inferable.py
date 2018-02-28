# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from pathlib import Path
from gensim.models.wrappers.fasttext import FastText
from models.CNN.intent_model.utils import download_untar


class EmbeddingInferableModel(object):

    def __init__(self, embedding_dim, embedding_fname=None, embedding_url=None,  *args, **kwargs):
        """
        Method initialize the class according to given parameters.
        Args:
            embedding_fname: name of file with embeddings
            embedding_dim: dimension of embeddings
            embedding_url: url link to embedding to try to download if file does not exist
            *args:
            **kwargs:
        """
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.model = None
        self.load(embedding_fname, embedding_url)

    def add_items(self, sentence_li):
        """
        Method adds new items to tok2emb dictionary from given text
        Args:
            sentence_li: list of sentences

        Returns: None

        """
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    try:
                        self.tok2emb[tok] = self.model[tok]
                    except KeyError:
                        self.tok2emb[tok] = np.zeros(self.embedding_dim)
        return

    def emb2str(self, vec):
        """
        Method returns string corresponding to the given embedding vectors
        Args:
            vec: vector of embeddings

        Returns:
            string corresponding to the given embeddings
        """
        string = ' '.join([str(el) for el in vec])
        return string

    def load(self, embedding_fname, embedding_url=None, *args, **kwargs):
        """
        Method initializes dict of embeddings from file
        Args:
            fname: file name

        Returns:
            Nothing
        """

        if not embedding_fname:
            raise RuntimeError('No pretrained fasttext intent_model provided')
        fasttext_model_file = embedding_fname

        if not Path(fasttext_model_file).is_file():
            emb_path = embedding_url
            if not emb_path:
                raise RuntimeError('No pretrained fasttext intent_model provided')
            embedding_fname = Path(fasttext_model_file).name
            try:
                download_path = './'
                download_untar(embedding_url, download_path)
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)
        self.model = FastText.load_fasttext_format(fasttext_model_file)
        return

    def infer(self, instance, *args, **kwargs):
        """
        Method returns embedded data
        Args:
            instance: sentence or list of sentences

        Returns:
            Embedded sentence or list of embedded sentences
        """
        if type(instance) is str:
            tokens = instance.split(" ")
            self.add_items(tokens)
            embedded_tokens = []
            for tok in tokens:
                embedded_tokens.append(self.tok2emb.get(tok))
            if len(tokens) == 1:
                embedded_tokens = embedded_tokens[0]
            return embedded_tokens
        else:
            embedded_instance = []
            for sample in instance:
                tokens = sample.split(" ")
                self.add_items(tokens)
                embedded_tokens = []
                for tok in tokens:
                    embedded_tokens.append(self.tok2emb.get(tok))
                embedded_instance.append(embedded_tokens)
            return embedded_instance

