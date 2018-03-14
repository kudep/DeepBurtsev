from deeppavlov.core.commands.infer import build_model_from_config
import json
import pandas as pd
from tqdm import tqdm


class Speller(object):

    def __init__(self, speller_conf=None):
        if speller_conf is None:
            self.conf_path = './DeepPavlov/deeppavlov/configs/error_model/brillmoore_kartaslov_ru.json'
        else:
            self.conf_path = speller_conf

        with open(self.conf_path) as config_file:
            self.config = json.load(config_file)

        self.speller = build_model_from_config(self.config)

    def transform(self, data):
        refactor = list()
        for x in tqdm(data['request']):
            refactor.append(self.speller([x])[0])

        df = pd.DataFrame({'request': refactor,
                           'class': data['class']})

        return df
