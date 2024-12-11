from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = scale(iris.data) #ปรับขนาดข้อมูลให้เป็นมาตรฐาน (Standardization) โดย: ค่าเฉลี่ยของแต่ละคอลัมน์ = 0, ส่วนเบี่ยงเบนมาตรฐาน = 1
#เป็น NumPy Array ขนาด 150 x 4 (150 ตัวอย่าง, 4 คุณลักษณะ)

Y = pd.DataFrame(iris.target) #ป้ายกำกับจริง (True Labels) ของข้อมูล Iris Dataset

clustering = KMeans(n_clusters=3, random_state=5) #สร้างโมเดล K-Means สำหรับการจัดกลุ่มข้อมูลเป็น 3 กลุ่ม (Clusters)
clustering.fit(X) #ฝึกโมเดล K-Means ด้วยข้อมูลคุณลักษณะ (Features) ที่ปรับขนาดแล้ว (X)
"""
ใช้เฉพาะ X ในการฝึกโมเดล K-Means เพราะมันเป็นวิธีการ Unsupervised Learning ที่ไม่ต้องการป้ายกำกับ (y)
หากคุณมีป้ายกำกับจริง (y), คุณสามารถใช้มันเพื่อตรวจสอบผลลัพธ์ของการจัดกลุ่มหลังจาก K-Means ทำงานเสร็จแล้ว.
"""
print(clustering.labels_) #แสดงกลุ่ม (Cluster) ที่ K-Means จัดให้กับแต่ละตัวอย่างในข้อมูล / เลข 0, 1, 2 หมายถึงกลุ่มที่ K-Means จัดไว้

print(Counter(clustering.labels_)) #ใช้ Counter เพื่อตรวจสอบจำนวนตัวอย่างในแต่ละกลุ่ม:

print(accuracy_score(Y, clustering.labels_)) # คุณสามารถเปรียบเทียบผลลัพธ์ของ .labels_ กับ iris.target เพื่อดูความถูกต้องของการจัดกลุ่ม

plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis') #ใช้ Scatter Plot เพื่อดูการจัดกลุ่มในเชิงภาพ
plt.title("K-Means Clustering")
plt.show()
