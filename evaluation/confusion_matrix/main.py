import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from eval import get_clf_eval
from roc_curve_plot import roc_curve_plot

# NULL 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩 수행
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + "/csv"
titanic_df = pd.read_csv(path + "/titanic" + "/train.csv") # data 불러오기


y_titanic_df = titanic_df['Survived'] # 정답 설정
X_titanic_df = titanic_df.drop('Survived', axis=1) # column 삭제
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train) # 학습
pred = lr_clf.predict(X_test) # 예측

pred_proba = lr_clf.predict_proba(X_test)[:, 1] # 개별 데이터별 예측 확률 반환

get_clf_eval(y_test, pred, pred_proba) # 평가

print("pred_proba() result shape: ", pred_proba.shape)
print("Just three sample pred_proba array: ", pred_proba[:3])

pred_proba = lr_clf.predict_proba(X_test) # 개별 데이터별 예측 확률 반환
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1) # 열에 추가
print('greater probability predict class value of two classes: ', pred_proba_result[:3])

custom_threshold_list = [0.4, 0.45, 0.5, 0.55, 0.6] # 임계값

for custom_threshold in custom_threshold_list:

    pred_proba_1 = pred_proba[:, 1].reshape(-1, 1) # positive column만 추출해서 binarizer적용

    binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) # 학습
    custom_predict = binarizer.transform(pred_proba_1)

    print("Threshold: ", custom_threshold)

    get_clf_eval(y_test, custom_predict)

pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1]

precision, recalls, threshold = precision_recall_curve(y_test, pred_proba_class1)
print("threshold shape: ", threshold.shape)

thr_index = np.arange(0, threshold.shape[0], 15)
print('threshold array index for extracting samples: ', thr_index)
print('threshold of 10 samples: ', np.round(threshold[thr_index], 2))
print('precision of each threshold sample: ', np.round(precision[thr_index], 3))
print('recall of each threshold sample: ', np.round(recalls[thr_index], 3))

plt.figure(figsize=(8,6))
threshold_boundary = threshold.shape[0]
plt.plot(threshold, precision[0:threshold_boundary], linestyle='--', label='precision')
plt.plot(threshold, recalls[0:threshold_boundary], label='recall')

start, end = plt.xlim()
plt.xticks(np.round(np.arange(start, end, 0.1), 2)) # x축 눈금 표시

plt.xlabel('Threshold value')
plt.ylabel('Precision and Recall value')
plt.legend()
plt.grid()
plt.show()

fprs, tprs, threshold = roc_curve(y_test, pred_proba_class1)
thr_index = np.arange(1, threshold.shape[0], 5)
print("threshold array's index for extracting sample: ", thr_index)
print("threshold value extracting sample index: ", np.round(threshold[thr_index], 2))

print("FPR value each sample threshold: ", np.round(fprs[thr_index], 3))
print("TPR value each sample threshold: ", np.round(tprs[thr_index], 3))

roc_curve_plot(y_test, pred_proba[:, 1])

pred_proba = lr_clf.predict_proba(X_test)[:, 1] # 개별 데이터별 예측 확률 반환
roc_score = roc_auc_score(y_test, pred_proba)
print(f"ROC AUC value: {roc_score}")