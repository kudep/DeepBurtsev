from deepburtsev.core.transformers import Tokenizer, Lemmatizer, TextConcat, Lower
from deepburtsev.core.pipegen import PipelineGenerator

struct = [[Tokenizer, Tokenizer], (Lemmatizer, {"op_name": "test_lem"}), (TextConcat, {"search": True,
                                                                                       "op_name": ['name1',
                                                                                                   'name2',
                                                                                                   'name3']}),
          Lower]


pipegen = PipelineGenerator(struct, n=3)

k = 0
for x in pipegen():
    print(x)
    k += 1

print(k)
