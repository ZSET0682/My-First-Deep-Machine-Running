from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# 데이터 불러오고 정리하기!
cancer = load_breast_cancer()
cancer_feature = cancer.feature_names  
cancer_target = cancer.target_names
cancer_data = cancer.data
cancer_label = cancer.target

print('유방암 진단 프로젝트!\n\n')
print('아래는 설명!(Describe)\n')
print(cancer.DESCR)
print('target_names : ',cancer_target, '\n')
print('feature_names : ',cancer_feature, '\n\n')

# sklearn에서 모델 5종류 임포트!
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# 모델 5종류 변수할당!
model_Tree = DecisionTreeClassifier(random_state=1)
model_RandomForest = RandomForestClassifier(random_state=1)
model_svm = svm.SVC(random_state=1)  
model_SGDClassifier = SGDClassifier(random_state=1)
model_LogisticRegreession = LogisticRegression(random_state=1,max_iter=2000)  """ LogisticRegreession은 기본 반복횟수로는 수렴하지 못했다..왜지? 아마도 데이터셋의 양 
때문인듯 한데 정확한 정리를 찾아볼 수 없었다. 
"""
model = [model_Tree,model_RandomForest,model_svm,model_SGDClassifier,model_LogisticRegreession]  # 모델들을 리스트에 넣어 반복문 준비하기!
for model in model:  # 모델 한꺼번에 학습시키기 & 결과 표시하기!
    print('사용한 모델 :',model,'\n\n') 
    X_train, X_test, y_train, y_test = train_test_split(cancer_data, cancer_label, test_size=0.2, random_state=2)  # 모델 8:2 로 분리하기

    model.fit(X_train, y_train)  # 답안과 정답지로 훈련!
    y_pred = model.predict(X_test)  # 학습이 완료된 데이터(X_train,y_train)를 기반으로 검증용 데이터의 정답예측
    print(classification_report(y_test, y_pred))  # 검증용 정답지(y_test) 와 비교

"""
유방암 여부 진단 데이터 셋은 양성(유방암)을 음성(정상)으로 판단하면 절대 안되는것으로 판단되어 Recall지표가 높은 
RandomForestClassifier 모델이 가장 적합해 보인다.
"""
