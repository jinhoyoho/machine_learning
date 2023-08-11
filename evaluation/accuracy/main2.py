from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X): # 입력값으로 들어오는 X를 모두 0으로 반환
        return np.zeros((len(X), 1), dtype=bool)
    
digits = load_digits()

y = (digits.target == 7).astype(int) # 7만 정답
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

print("Label test size: ", y_test.shape)
print("Distribution 0 and 1 of test set: ", pd.Series(y_test).value_counts()) # pandas 벡터 생성, 발생 횟수 반환

fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train) # 학습(pass)
fakepred = fakeclf.predict(X_test) # 예측
print("Accuracy that all predict 0: {:.3f}".format(accuracy_score(y_test, fakepred))) # 정확도 출력