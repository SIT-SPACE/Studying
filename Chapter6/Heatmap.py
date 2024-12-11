from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris() #โหลด Iris Dataset ซึ่งเป็นชุดข้อมูลที่ประกอบด้วยข้อมูลเกี่ยวกับดอกไม้ประเภท Iris  มีคุณสมบัติ 4 ตัว เช่น sepal_length มีข้อมูลทั้งหมดเลย
datasets = iris.data # เอาแค่ iris.data: ข้อมูลคุณลักษณะ (features) ซึ่งยังเป็น array อยู่

df = pd.DataFrame(datasets, columns=iris.feature_names) #โดยกำหนดชื่อคอลัมน์จาก iris.feature_names 

corrMatrix = df.corr() # คำนวณค่าสหสมพันธ์ระหว่างคอลัม

sns.heatmap(corrMatrix, annot=True) #แสดงค่าตัวเลขในแต่ละเซลล์ของ Heatmap
plt.show()