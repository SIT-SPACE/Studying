import pandas as pd
from sklearn import tree # ใช้สร้างและฝึกโมเดล Decision Tree Classifier
from sklearn import datasets
import graphviz
from sklearn import metrics # ใช้คำนวณ accuracy, precision, recall, F1 score และ Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

HOW_DEEP_TREE = 5 # มีทั้งหมด 32 Leaf Nodes

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=50)

clf = tree.DecisionTreeClassifier(random_state=30, max_depth=HOW_DEEP_TREE)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))
# คำนวณ accuracy หรือ ความแม่นยำ ของโมเดลระหว่าง
# y_test: ค่าจริงในชุดทดสอบ / y_pred: ค่าที่โมเดลพยากรณ์

cm = pd.DataFrame(confusion_matrix(y_test,y_pred), columns=iris.target_names,
                  index=iris.target_names)
'''
Confusion Matrix ใช้แสดงการทำนายของโมเดล โดยการเปรียบเทียบ ค่าจริง (y_test) และ ค่าที่ทำนาย (y_pred)
มันจะสร้างเมทริกซ์ที่มี 4 ช่อง:
True Positives (TP): จำนวนครั้งที่โมเดลทำนายถูกต้อง (เช่น ทำนายว่าเป็น setosa และมันคือ setosa)
False Positives (FP): จำนวนครั้งที่โมเดลทำนายผิด (เช่น ทำนายว่าเป็น setosa แต่จริง ๆ มันเป็น versicolor)
True Negatives (TN): จำนวนครั้งที่โมเดลทำนายถูกต้องว่าเป็น not setosa
False Negatives (FN): จำนวนครั้งที่โมเดลทำนายผิดว่าไม่เป็น setosa
'''
print(cm)