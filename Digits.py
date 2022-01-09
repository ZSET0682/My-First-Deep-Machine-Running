from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import pandas as pd

digits = load_digits()
digits_feature = digits.feature_names
digits_target = digits.target_names
digits_data = digits.data
digits_label = digits.target
print('손글씨 분류 프로젝트!\n')
print('아래는 설명!(Describe)\n')
print(digits.DESCR)
print('target_names : ',digits_target, '\n')
print('feature_names :', digits_feature, '\n\n')

model_Tree = DecisionTreeClassifier(random_state=1)
model_RandomForest = RandomForestClassifier(random_state=1)
model_svm = svm.SVC(random_state=1)
model_SGDClassifier = SGDClassifier(random_state=1)
model_LogisticRegreession = LogisticRegression(random_state=1,max_iter=2000)  # LogisticRegreession은 기본 반복횟수로는 수렴하지 못했다..

model = [model_Tree,model_RandomForest,model_svm,model_SGDClassifier,model_LogisticRegreession]
for model in model:
    print('사용한 모델 :',model,'\n\n')
    X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                        digits_label, 
                                                        test_size=0.2, 
                                                        random_state=1)

    model.fit(X_train, y_train)  # 답안과 정답지로 훈련!
    y_pred = model.predict(X_test)  # 학습이 완료된 데이터(X_train,y_train)를 기반으로 검증용 데이터의 정답예측
    print(classification_report(y_test, y_pred))  # 검증용 정답지(y_test) 와 비교
'''
손글씨 분류 프로젝트에서는 정답이 아닌 글자(음성)를 정답인 글자(음성)로 표시하게되면 프로젝트 목적에 부합하기 힘들기 때문에 Precision 수치가 제일 높은
SVC 모델이 가장 적합해 보인다.
'''