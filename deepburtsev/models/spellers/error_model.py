import csv
import itertools
from collections import defaultdict, Counter
from heapq import heappop, heappushpop, heappush
from math import log, exp

import kenlm
from tqdm import tqdm

from .utils import expand_path
from .utils import ConfigError
from .utils import StaticDictionary


# need to create dictionary
class ErrorModel(object):
    def __init__(self, dictionary: StaticDictionary, window=1, lm_file=None, save_path="error_model/error_model_ru.tsv",
                 load_path="error_model/error_model_ru.tsv"):

        # super().__init__(*args, **kwargs)

        self.load_path = expand_path(load_path)
        self.save_path = expand_path(save_path)

        self.costs = defaultdict(itertools.repeat(float('-inf')).__next__)
        self.dictionary = dictionary
        self.window = window
        if self.window == 0:
            self.find_candidates = self._find_candidates_window_0
        else:
            self.find_candidates = self._find_candidates_window_n
        self.costs[('', '')] = log(1)
        self.costs[('⟬', '⟬')] = log(1)
        self.costs[('⟭', '⟭')] = log(1)

        for c in self.dictionary.alphabet:
            self.costs[(c, c)] = log(1)
        # if self.ser_path.is_file():
        self.load()

        if lm_file:
            self.lm = kenlm.Model(str(expand_path(lm_file)))
            self.beam_size = 4
            self.candidates_count = 4
            self._infer_instance = self._infer_instance_lm

    def _find_candidates_window_0(self, word, k=1, prop_threshold=1e-6):
        threshold = log(prop_threshold)
        d = {}
        prefixes_heap = [(0, {''})]
        candidates = [(float('-inf'), '') for _ in range(k)]
        word = '⟬{}⟭'.format(word.lower().replace('ё', 'е'))
        word_len = len(word) + 1
        while prefixes_heap and -prefixes_heap[0][0] > candidates[0][0]:
            _, prefixes = heappop(prefixes_heap)
            for prefix in prefixes:
                res = []
                for i in range(word_len):
                    c = word[i - 1:i]
                    res.append(max(
                        (res[-1] + self.costs[('', c)]) if i else float('-inf'),
                        d[prefix[:-1]][i] + self.costs[(prefix[-1], '')] if prefix else float(
                            '-inf'),
                        (d[prefix[:-1]][i - 1] + (self.costs[(prefix[-1], c)]))
                        if prefix and i else float('-inf')
                    ) if i or prefix else 0)
                d[prefix] = res
                if prefix in self.dictionary.words_set:
                    heappushpop(candidates, (res[-1], prefix))
                potential = max(res)
                if potential > threshold:
                    heappush(prefixes_heap, (-potential, self.dictionary.words_trie[prefix]))
        return [(w.strip('⟬⟭'), score) for score, w in sorted(candidates, reverse=True) if
                score > threshold]

    def _find_candidates_window_n(self, word, k=1, prop_threshold=1e-6):
        threshold = log(prop_threshold)
        word = '⟬{}⟭'.format(word.lower().replace('ё', 'е'))
        word_len = len(word) + 1
        inf = float('-inf')
        d = defaultdict(list)
        d[''] = [0.] + [inf] * (word_len - 1)
        prefixes_heap = [(0, self.dictionary.words_trie[''])]
        candidates = [(inf, '')] * k
        while prefixes_heap and -prefixes_heap[0][0] > candidates[0][0]:
            _, prefixes = heappop(prefixes_heap)
            for prefix in prefixes:
                prefix_len = len(prefix)
                d[prefix] = res = [inf]
                for i in range(1, word_len):
                    c_res = [inf]
                    for li in range(1, min(prefix_len + 1, self.window + 2)):
                        for ri in range(1, min(i + 1, self.window + 2)):
                            prev = d[prefix[:-li]][i - ri]
                            if prev > threshold:
                                edit = (prefix[-li:], word[i - ri:i])
                                if edit in self.costs:
                                    c_res.append(prev +
                                                 self.costs[edit])
                    res.append(max(c_res))
                if prefix in self.dictionary.words_set:
                    heappushpop(candidates, (res[-1], prefix))
                potential = max(res)
                # potential = max(
                #     [e for i in range(self.window + 2) for e in d[prefix[:prefix_len - i]]])
                if potential > threshold:
                    heappush(prefixes_heap, (-potential, self.dictionary.words_trie[prefix]))
        return [(w.strip('⟬⟭'), score) for score, w in sorted(candidates, reverse=True) if
                score > threshold]

    def _infer_instance(self, instance: str):
        corrected = []

        # old version
        # for incorrect in instance.split():
        #     if any([c not in self.dictionary.alphabet for c in incorrect]):
        #         corrected.append(incorrect)
        #     else:
        #         res = self.find_candidates(incorrect, k=1, prop_threshold=1e-6)
        #         corrected.append(res[0][0] if res else incorrect)

        if isinstance(instance, str):
            for incorrect in instance.split():
                if any([c not in self.dictionary.alphabet for c in incorrect]):
                    corrected.append(incorrect)
                else:
                    res = self.find_candidates(incorrect, k=1, prop_threshold=1e-6)
                    corrected.append(res[0][0] if res else incorrect)
        else:
            for incorrect in instance:
                if any([c not in self.dictionary.alphabet for c in incorrect]):
                    corrected.append(incorrect)
                else:
                    res = self.find_candidates(incorrect, k=1, prop_threshold=1e-6)
                    corrected.append(res[0][0] if res else incorrect)
        return ' '.join(corrected)

    def _infer_instance_lm(self, instance: str):
        candidates = []

        # old version
        # for incorrect in instance.split():
        #     if any([c not in self.dictionary.alphabet for c in incorrect]):
        #         candidates.append([(0, incorrect)])
        #     else:
        #         res = self.find_candidates(incorrect, k=self.candidates_count, prop_threshold=1e-6)
        #         if res:
        #             candidates.append([(score, candidate) for candidate, score in res])
        #         else:
        #             candidates.append([(0, incorrect)])
        # candidates.append([(0, '</s>')])

        # new version
        if isinstance(instance, str):
            for incorrect in instance.split():
                if any([c not in self.dictionary.alphabet for c in incorrect]):
                    candidates.append([(0, incorrect)])
                else:
                    res = self.find_candidates(incorrect, k=self.candidates_count, prop_threshold=1e-6)
                    if res:
                        candidates.append([(score, candidate) for candidate, score in res])
                    else:
                        candidates.append([(0, incorrect)])
            candidates.append([(0, '</s>')])
        else:
            for incorrect in instance:
                if any([c not in self.dictionary.alphabet for c in incorrect]):
                    candidates.append([(0, incorrect)])
                else:
                    res = self.find_candidates(incorrect, k=self.candidates_count, prop_threshold=1e-6)
                    if res:
                        candidates.append([(score, candidate) for candidate, score in res])
                    else:
                        candidates.append([(0, incorrect)])
            candidates.append([(0, '</s>')])

        # next
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        beam = [(0, state, [])]
        for sublist in candidates:
            new_beam = []
            for beam_score, beam_state, beam_words in beam:
                for score, candidate in sublist:
                    state = kenlm.State()
                    c_score = self.lm.BaseScore(beam_state, candidate, state)
                    new_beam.append((beam_score + score + c_score, state, beam_words + [candidate]))
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        score, state, words = beam[0]
        return ' '.join(words[:-1])

    def __call__(self, data, *args, **kwargs):
        if isinstance(data, str):
            return self._infer_instance(data)
        if len(data) > 1:
            data = tqdm(data, desc='Infering a batch with the error model', leave=False)
        return [self._infer_instance(instance) for instance in data]

    def reset(self):
        pass

    @staticmethod
    def _distance_edits(seq1, seq2):
        l1, l2 = len(seq1), len(seq2)
        d = [[(i, ()) for i in range(l2 + 1)]]
        d += [[(i, ())] + [(0, ())] * l2 for i in range(1, l1 + 1)]

        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                edits = [
                    (d[i - 1][j][0] + 1, d[i - 1][j][1] + ((seq1[i - 1], ''),)),
                    (d[i][j - 1][0] + 1, d[i][j - 1][1] + (('', seq2[j - 1]),)),
                    (d[i - 1][j - 1][0] + (seq1[i - 1] != seq2[j - 1]),
                     d[i - 1][j - 1][1] + ((seq1[i - 1], seq2[j - 1]),))
                ]
                if i > 1 and j > 1 and seq1[i - 1] == seq2[j - 2] and seq1[i - 2] == seq2[j - 1]:
                    edits.append((d[i - 2][j - 2][0] + (seq1[i - 1] != seq2[j - 1]),
                                  d[i - 2][j - 2][1] + ((seq1[i - 2:i], seq2[j - 2:j]),)))
                d[i][j] = min(edits, key=lambda x: x[0])

        return d[-1][-1]

    def fit(self, x, y):
        changes = []
        entries = []
        data = list(zip(x, y))
        window = 4
        for error, correct in tqdm(data, desc='Training the error model'):
            correct = '⟬{}⟭'.format(correct)
            error = '⟬{}⟭'.format(error)
            d, ops = self._distance_edits(correct, error)
            if d <= 2:
                w_ops = set()
                for pos in range(len(ops)):
                    left, right = list(zip(*ops))
                    for l in range(pos, max(0, pos - window) - 1, -1):
                        for r in range(pos + 1, min(len(ops), l + 2 + window)):
                            w_ops.add(((''.join(left[l:r]), ''.join(right[l:r])), l, r))
                ops = [x[0] for x in w_ops]

                entries += [op[0] for op in ops]
                changes += [op for op in ops]

        e_count = Counter(entries)
        c_count = Counter(changes)
        incorrect_prior = 1
        correct_prior = 19
        for (w, s), c in c_count.items():
            c = c + (incorrect_prior if w != s else correct_prior)
            e = e_count[w] + incorrect_prior + correct_prior
            p = c / e
            self.costs[(w, s)] = log(p)

    def save(self):
        print("[saving error_model to `{}`]".format(self.save_path))

        with open(self.save_path, 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for (w, s), log_p in self.costs.items():
                writer.writerow([w, s, exp(log_p)])

    def load(self):
        if self.load_path:
            if self.load_path.is_file():
                print("loading error_model from `{}`".format(self.load_path))
                with open(self.load_path, 'r', newline='') as tsv_file:
                    reader = csv.reader(tsv_file, delimiter='\t')
                    for w, s, p in reader:
                        self.costs[(w, s)] = log(float(p))
            elif not self.load_path.parent.is_dir():
                raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                    self.__class__.__name__))
        else:
            print('No load_path provided, initializing error model from scratch')
