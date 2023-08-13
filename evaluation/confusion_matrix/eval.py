from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred) # 오차행렬
    accuracy = accuracy_score(y_test, pred) # 정확도
    precision = precision_score(y_test, pred) # 정밀도
    recall = recall_score(y_test, pred) # 재현율
    f1 = f1_score(y_test, pred) # f1 score
    
    roc_auc = roc_auc_score(y_test, pred_proba)

    print("Confusion Matrix:")
    print(confusion)

    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, AUC: {roc_auc}")
    