import bentoml

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma="scale")
clf.fit(X, y)

saved_model = bentoml.sklearn.save_model("iris_classifier", clf)
print("Model saved: ", saved_model)

# iris_classifier:ug63mtebiklwjedf
