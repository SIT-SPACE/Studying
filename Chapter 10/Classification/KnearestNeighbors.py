from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
iris = datasets.load_iris()

x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors=5) # เพื่อนบ้าน 5 จุดที่ใกล้ที่สุดในการตัดสินใจว่าตัวอย่างใหม่เป็นกลุ่มใด

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_pred,y_test))

print("y_test: ",y_test)
print("y_pred:", y_pred)

