from sklearn import datasets #Scikit-learn: ใช้สำหรับโหลดหรือสร้างชุดข้อมูล
import numpy as np #NumPy: ใช้จัดการข้อมูลในรูปแบบอาร์เรย์และการคำนวณ
import pandas as pd #Pandas: ใช้จัดการข้อมูลในรูปแบบตาราง (DataFrame)
import matplotlib.pyplot as plt #Matplotlib: ใช้สร้างกราฟเพื่อวิเคราะห์และแสดงผลข้อมูล

data = pd.read_csv('D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/iris.csv') #โหลดข้อมูล
df = pd.DataFrame(data) #แปลงข้อมูลเป็น DataFrame โดยใช้ไลบารี่ pandas

data_histogram1 = df["sepal_length"] # ดึงข้อมูลจากคอลัม sepal_length
m1 = [m1 for m1 in range(len(data_histogram1))] #สร้างลำดับของข้อมูล
plt.bar(m1, data_histogram1.values) #สร้างกราฟแท่ง (ลำดับ (แกน x), ดึงค่าข้อมูลของ sepal_length(แกน Y) )
plt.title("Sepal Length") #ตั้งชื่อกราฟ
plt.show()

data_histogram2 = df["sepal_width"]
m2 = [m2 for m2 in range(len(data_histogram2))]
plt.bar(m2, data_histogram2)
plt.title("Sepal Width")
plt.show()

data_histogram3 = df["petal_length"]
m3 = [m3 for m3 in range(len(data_histogram3))]
plt.bar(m3, data_histogram3)
plt.title("Petal Length")
plt.show()

data_histogram4 = df["petal_width"]
m4 = [m4 for m4 in range(len(data_histogram4))]
plt.bar(m4, data_histogram4)
plt.title("Petal Width")
plt.show()

plt.hist(data_histogram4.values, cumulative=True)
plt.title("Cumulative Histogram")
plt.show()