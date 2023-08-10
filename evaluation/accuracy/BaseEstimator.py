from sklearn.base import BaseEstimator
import numpy as np

class MyDummyClassifier(BaseEstimator):
    def fit(self, X, y=None): # 아무것도 학습하지 않음
        pass

    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))

        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:  # i번째 행에 있는 값
                pred[i] = 0
            else:
                pred[i] = 1

        return pred