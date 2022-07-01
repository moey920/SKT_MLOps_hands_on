from sklearn.datasets import load_iris
import pandas as pd

# 학습을 위한 라이브러리 임포트
from sklearn.linear_model import LogisticRegression  # Logistic(Regression)Classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.naive_bayes import GaussianNB  # Naive Bayesian
from sklearn.neighbors import KNeighborsClassifier  # K Nearest Neighbor
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosing
from sklearn.neural_network import MLPClassifier  # Neural Network

from sklearn.metrics import accuracy_score

from sklearn import model_selection

iris = load_iris()  # sample data load

# 로드된 데이터가 속성-스타일 접근을 제공하는 딕셔너리와 번치 객체로 표현된 것을 확인
print(iris)
# Description 속성을 이용해서 데이터셋의 정보를 확인
print(iris.DESCR)

# 각 key에 저장된 value 확인
print(iris.data)
# feature
print(iris.feature_names)

# label
print(iris.target)
print(iris.target_names)

# feature_names 와 target을 레코드로 갖는 데이터프레임 생성
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 0.0, 1.0, 2.0으로 표현된 label을 문자열로 매핑
df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
print(df)

# 슬라이싱을 통해 feature와 label 분리
x_data = df.iloc[:, :-1]
y_data = df.iloc[:, [-1]]

# logistic (Regression) Classifier, Decision tree, support vector machine, naive bayesian, K Nearest Neighbor, Random Forest, Gradient Boosing, Neural Network
models = []
models.append(("LR", LogisticRegression()))
models.append(("DT", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
models.append(("NB", GaussianNB()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("GB", GradientBoostingClassifier()))
models.append(("ANN", MLPClassifier()))

# 모델 학습 및 정확도 분석
for name, model in models:
    model.fit(x_data, y_data.values.flatten())
    y_pred = model.predict(x_data)
    print(name, "'s Accuracy is ", accuracy_score(y_data, y_pred))


"""sklearn의 model_selection을 활용한 하이퍼 파라미터 최적화도 가능하지만, Katib을 사용하도록합니다.
model = LogisticRegression()
parameters = {
    'C' : [2**0, 2**3, 2**6, 2**9, 2**12],
    'random_state' : [0, 7, 13, 42]
}
gs = model_selection.GridSearchCV(model, parameters)
gs.fit(x_data, y_data.values.ravel())
model = gs.best_estimator_

kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
cv_results = model_selection.cross_val_score(model, x_data, y_data.values.ravel(), cv=kfold, scoring="accuracy")
print(cv_results)
"""
