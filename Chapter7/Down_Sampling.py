from sklearn.datasets import make_classification
from collections import Counter
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


X,y = make_classification(n_classes=2, n_redundant=0, weights=[0.1,0.9],
                          n_features=2, n_samples=500, random_state=4)

df = pd.DataFrame(X) #สร้าง DataFrame จากข้อมูลคุณลักษณะใน X
df["Class"] = y #เพิ่มคอลัมน์ Class เพื่อเก็บป้ายกำกับจาก y
print(Counter(df["Class"])) #ใช้นับจำนวนตัวอย่างในแต่ละคลาส

df_minority = df[df.iloc[:,2].values==0] #เลือกเฉพาะแถวที่คลาสเป็น 0 (Minority Class)
df_majority = df[df.iloc[:,2].values==1] #เลือกเฉพาะแถวที่คลาสเป็น 1 (Majority Class)
df_majority_downsampled = resample(df_majority, #ใช้สุ่มตัวอย่างใหม่จากคลาสที่มีมากกว่า (Majority Class)
                                    n_samples=53, #n_samples=53: กำหนดจำนวนตัวอย่างหลังการ Downsampling ให้เท่ากับจำนวนตัวอย่างใน Minority Class (50 หรือใกล้เคียง)
                                    random_state=123)
#df_majority_downsampled คือ DataFrame ที่ลดจำนวนตัวอย่างใน Majority Class

df_downsampling = pd.concat([df_majority_downsampled,
                            df_minority])
print(df_downsampling) #รวมข้อมูลของ Majority Class หลังการ Downsampling กับ Minority Class

colors = np.array(['blue', 'red'])

plt.subplot(1,2,1)
plt.scatter(df.iloc[:,0], df.iloc[:,1],
            color=colors[df['Class']], linewidths=0.1)
plt.title("Original Data")

plt.subplot(1,2,2)
plt.scatter(df_downsampling.iloc[:,0],
            df_downsampling.iloc[:,1],
            color=colors[df_downsampling['Class']], linewidths=0.1)
plt.title("Downsampling")
plt.show()
