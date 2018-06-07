from typing import Generator
import random


class Dataset(object):

    def __init__(self, data, seed=None, classes_description=None):

        self.main_names = ['x', 'y']
        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.data = data

        self.classes_description = classes_description

    def iter_batch(self, batch_size: int, data_type: str = 'base', shuffle: bool = True,
                   only_request: bool = False) -> Generator:
        """This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): shuffle trigger
            only_request (bool): trigger that told what data will be returned
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data['x'])
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        if shuffle:
            random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        # for i in range((data_len - 1) // batch_size + 1):
        #     yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))
        if not only_request:
            for i in range((data_len - 1) // batch_size + 1):
                # o = order[i * batch_size: (i + 1) * batch_size]
                # print(type(o))
                # print(o)

                yield list((list(data[self.main_names[0]][i * batch_size: (i + 1) * batch_size]),
                            list(data[self.main_names[1]][i * batch_size: (i + 1) * batch_size])))
        else:
            for i in range((data_len - 1) // batch_size + 1):
                o = order[i * batch_size:(i + 1) * batch_size]
                yield list((list(self.data[self.main_names[0]][o]),))

    def iter_all(self, data_type: str = 'base', only_request: bool = False) -> Generator:
        """
        Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
            only_request (bool): trigger that told what data will be returned
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in zip(data[self.main_names[0]], data[self.main_names[1]]):
            if not only_request:
                yield (x, y)
            else:
                yield (x,)
