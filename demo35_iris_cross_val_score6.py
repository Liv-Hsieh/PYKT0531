import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target

logisticRegression1 = LogisticRegression()
svc1 = SVC()
tree1 = DecisionTreeClassifier()
# randomForest1 = RandomForestClassifier(n_estimators=100, oob_score=True)
randomForest1 = RandomForestClassifier(n_estimators=200, oob_score=True)
knn1 = KNeighborsClassifier(n_neighbors=4)

classifiers = [logisticRegression1, svc1, tree1, randomForest1, knn1]

for c in classifiers:
    score = model_selection.cross_val_score(c, X, Y, cv=3)
    print(f"classifier:{c} has score:{score}")
