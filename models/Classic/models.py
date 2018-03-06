from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from utils import get_result


# TODO write test method in all classes
class LinearRegression(object):
    def __init__(self, vectorizer, train=True, *args):
        self.model = LogisticRegression(n_jobs=-1, solver='lbfgs')
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, data):
        if self.train:
            vec = self.vectorizer.fit_transform(data[0])
        else:
            vec = self.vectorizer.transform(data[0])
        self.model.fit(vec, data[1])

    def test(self, data):
        vec = self.vectorizer.transform(data[0])
        y_pred = self.model.predict(vec)
        result = get_result(y_pred, data[1], )

    def params(self):
        return self.model.get_params()


class RandomForest(object):
    def __init__(self, vectorizer, train=True, *args):
        self.model = RandomForestClassifier(max_depth=5, random_state=0)
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, data):
        if self.train:
            vec = self.vectorizer.fit_transform(data[0])
        else:
            vec = self.vectorizer.transform(data[0])
        self.model.fit(vec, data[1])

    def params(self):
        return self.model.get_params()


class SVM(object):
    def __init__(self, vectorizer, train=True, *args):
        self.model = LinearSVC(C=0.8873076204728344, class_weight='balanced')
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, data):
        if self.train:
            vec = self.vectorizer.fit_transform(data[0])
        else:
            vec = self.vectorizer.transform(data[0])
        self.model.fit(vec, data[1])

    def params(self):
        return self.model.get_params()


class GBM(object):
    def __init__(self, vectorizer, train=True, *args):
        self.model = LGBMClassifier(n_estimators=200, n_jobs=-1, learning_rate=0.1)
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, data):
        if self.train:
            vec = self.vectorizer.fit_transform(data[0])
        else:
            vec = self.vectorizer.transform(data[0])
        self.model.fit(vec, data[1])

    def params(self):
        return self.model.get_params()




