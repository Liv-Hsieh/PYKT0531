import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data
Y = iris.target

logisticRegression1 = LogisticRegression()

classifiers = [logisticRegression1]

for c in classifiers:
    score = model_selection.cross_val_score(logisticRegression1, X, Y, cv=3)
    print(f"classifier:{c} has score:{score}")