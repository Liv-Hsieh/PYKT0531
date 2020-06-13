import seaborn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
print(df)
seaborn.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.3, stratify=iris.target)
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)

predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f'oob score estimate={rf.oob_score_}')
print(f"mean accuracy for unknown:{accuracy}")

cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names,
                  index=iris.target_names)
print(cm)
seaborn.heatmap(cm, annot=True)
plt.show()