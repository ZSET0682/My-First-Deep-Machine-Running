from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

wine = load_wine()
wine_feature = wine.feature_names
wine_target = wine.target_names
wine_data = wine.data
wine_label = wine.target
print('와인 분류 프로젝트!\n')
print('아래는 설명!(Describe)\n')
print(wine.DESCR)
print('target_names : ',wine_target, '\n')
print('feature_names :', wine_feature, '\n\n')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
model_Tree = DecisionTreeClassifier(random_state=1)
model_RandomForest = RandomForestClassifier(random_state=1)
model_svm = svm.SVC(random_state=1)
model_SGDClassifier = SGDClassifier(random_state=1)
model_LogisticRegreession = LogisticRegression(random_state=1,max_iter=2500)  # LogisticRegreession은 기본 반복횟수로는 수렴하지 못했다..

model = [model_Tree,model_RandomForest,model_svm,model_SGDClassifier,model_LogisticRegreession]
for model in model:
    print('사용한 모델 :',model,'\n')
    X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                        wine_label, 
                                                        test_size=0.2, 
                                                        random_state=2)

    model.fit(X_train, y_train)  # 답안과 정답지로 훈련!
    y_pred = model.predict(X_test)  # 학습이 완료된 데이터(X_train,y_train)를 기반으로 검증용 데이터의 정답예측
    print(classification_report(y_test, y_pred))  # 검증용 정답지(y_test) 와 비교
"""
와인 분류 프로젝트에서는 예 또는 아니오 로 학습하는것이 아니고,물건의 재고가 하나밖에 없다고 해서 틀린 모델이 되는것이 아니기때문에
F1 Score 또는 Accuracy 가 가장 높은 모델인 RandomForestClassifier 모델이 적합할것 같다.
"""
