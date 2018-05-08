from deepburtsev.core.transformers import Tokenizer, Lemmatizer, TextConcat, Lower
from deepburtsev.core.pipelinemanager import PipelineGenerator

struct = [[Tokenizer, Tokenizer], (Lemmatizer, {"op_name": "test_lem"}), (TextConcat, {"search": True,
                                                                                       "op_name": ['name1',
                                                                                                   'name2',
                                                                                                   'name3']}),
          Lower]


info = {'name': "test",
        'root': './'}

path = '../../data/'
gen = PipelineGenerator(struct)
print(gen.length)
