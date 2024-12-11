from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

X,y = make_classification(n_classes=2, n_redundant=0, weights=[0.1,0.9],
                          n_features=2, n_samples=500, random_state=4)
# X: ข้อมูลคุณลักษณะ (Features) หรือค่าข้อมูล ในรูปแบบ NumPy Array (500 x 2)
# y: ค่าป้ายกำกับ (Labels) ซึ่งมีค่าทั้งหมดอยู่ในคลาส 0 หรือ 1

print("Before:", Counter(y).values())
print("Before:", Counter(y))

sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(X,y) #สร้างข้อมูลใหม่ที่สมดุลระหว่างคลาส
print("After:", Counter(y_resampled).values())
print("After:", Counter(y_resampled))

plt.scatter(X_resampled[:,0], X_resampled[:,1], 
            color='blue', linewidths=0.1, label="Upsamping")
plt.scatter(X[:,0], X[:,1],
            color='orange', linewidths=0.1, label="Original")
# แกน x และ y: คุณลักษณะที่ 1 และ 2 ของข้อมูล (จาก X[:, 0] และ X[:, 1])

plt.legend()
plt.show()
