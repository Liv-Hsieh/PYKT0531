import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df1 = pd.read_csv("data\\sonar.all-data", header=None, prefix='X')
print(df1.shape)
print(df1.columns)
data, labels = df1.iloc[:, :-1], df1.iloc[:, -1]
print(data.shape)
print(labels.shape)
df1.rename(columns={'X60': 'Label'}, inplace=True)
print(df1.columns)

classifier1 = KNeighborsClassifier(n_neighbors=3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
classifier1.fit(X_train, y_train)
y_predict = classifier1.predict(X_test)
print(f"score for test data:{classifier1.score(X_test, y_test)}")

result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(classifier1, data, labels, cv=5, groups=labels)
print(scores)