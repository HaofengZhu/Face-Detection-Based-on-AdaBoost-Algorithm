import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier=DecisionTreeClassifier,n_weakers_limit=5):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.classifiers = []
        self.alphas = []
    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        X=np.array(X)
        n = X.shape[0]
        M = self.n_weakers_limit
        w_m = np.array([1 / n] * n)
        for m in range(M):
            #e_m为训练集的分类误差率
            #w_m为每个sample的权值
            #alpha_m为每个分类器的权重
            classifier_m = self.classifier()
            classifier_m.fit(X,y,sample_weight=w_m)
            y_predict=classifier_m.predict(X)
            loss=0
            for i in range(len(y_predict)):
                if y_predict[i] != y[i]:
                    loss += 1
            if loss==0:
                self.classifiers.append(classifier_m)
                sum=0
                for a in self.alphas:
                    sum+=a
                self.alphas.append(1-sum)
                break
            e_m = loss * w_m[m]
            alpha_m = 1 / 2 * np.log((1 - e_m) / e_m)
            w_m = w_m * np.exp(-alpha_m * y * classifier_m.predict(X))
            z_m = np.sum(w_m)
            w_m = w_m / z_m
            self.classifiers.append(classifier_m)
            self.alphas.append(alpha_m)


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        X=np.array(X)
        n_samples, n_features = X.shape
        results = np.zeros((1,n_samples))
        for alpha, classifier in zip(self.alphas, self.classifiers):
            y=classifier.predict(X)
            y=np.array(y)
            results += alpha * y
        return results[0]


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        results = self.predict_scores(X)
        for i in range(len(results)):
            results[i]=((results[i] > 0) - 0.5) * 2
        return results

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
