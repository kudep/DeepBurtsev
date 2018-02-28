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

import nltk


class NLTKTokenizer(object):

    def infer(self, batch, tokenizer="wordpunct_tokenize"):
        tokenized_batch = []

        tokenizer_ = getattr(nltk.tokenize, tokenizer, None)
        if callable(tokenizer_):
            if type(batch) == str:
                tokenized_batch = " ".join(tokenizer_(batch))
            else:
                # list of str
                for text in batch:
                    tokenized_batch.append(" ".join(tokenizer_(text)))
            return tokenized_batch
        else:
            raise AttributeError("Tokenizer %s is not defined in nltk.tokenizer" % tokenizer)


