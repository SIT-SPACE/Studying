from sklearn.preprocessing import LabelEncoder # LabelEncoder: ใช้แปลงข้อมูลประเภทข้อความ (Categorical) เป็นตัวเลข
from sklearn.model_selection import train_test_split # train_test_split: ใช้แบ่งข้อมูลออกเป็นชุดฝึก (Train) และชุดทดสอบ (Test)
from sklearn.linear_model import LinearRegression # LinearRegression: โมเดลสำหรับการพยากรณ์ด้วย Linear Regression
from sklearn.metrics import mean_squared_error  #mean_squared_error, mean_absolute_error: ใช้วัดความแม่นยำของโมเดล
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/startup.csv')
# เริ่มจากคอลัมน์แรก (Index = 0) เลือกจนถึง คอลัมน์ก่อนสุดท้าย
X = dataset.iloc[:,:-1].values # X: ข้อมูลคุณลักษณะ เช่น R&D Spend, Administration, Marketing Spend, State
Y = dataset.iloc[:,4].values # Y: ตัวแปรเป้าหมาย (Profit)

labelencoder_X = LabelEncoder() # ใช้แปลงข้อมูลในคอลัมน์ State (เช่น New York, California, Florida) เป็นตัวเลข

X[:,3] = labelencoder_X.fit_transform(X[:,3]) #fit: เรียนรู้ค่าที่ไม่ซ้ำกันในคอลัมน์ (เช่น New York, California, Florida) / transform: แปลงค่าที่เรียนรู้แล้วให้เป็นตัวเลข

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#ชุดฝึก (Train Set): ใช้ในการฝึกโมเดล 80% / ชุดทดสอบ (Test Set): ใช้ตรวจสอบความแม่นยำของโมเดล 20%

regressor = LinearRegression() # เป็นการสร้าง ออบเจ็กต์ (Object) ของคลาส LinearRegression ซึ่งพร้อมใช้งาน แต่ยังไม่ได้ "ฝึก" โมเดลด้วยข้อมูล
'''
regressor คือ ออบเจ็กต์ (Object) ที่สร้างจากคลาส LinearRegression
เมื่อสร้างออบเจ็กต์แล้ว คุณสามารถใช้เมธอด (Methods) หรือแอตทริบิวต์ (Attributes) ต่างๆ ของคลาส LinearRegression ผ่านออบเจ็กต์นี้
'''

regressor.fit(X_train, y_train) # คำสั่งนี้เป็นการ "ฝึก" โมเดลโดยใช้ข้อมูลชุดฝึก (X_train, y_train)

# หลังจากโมเดลถูกฝึก (fit) และเรียนรู้ความสัมพันธ์ระหว่าง X_train และ y_train แล้ว เราใช้คำสั่ง predict() 
# เพื่อใช้โมเดลที่ฝึกไว้ พยากรณ์ค่าของเป้าหมาย (Target) สำหรับข้อมูลใหม่ (X_test)

y_pred = regressor.predict(X_test) # เป็นชุดข้อมูลที่ไม่ได้ใช้ในการฝึกโมเดลใช้สำหรับตรวจสอบว่าโมเดลสามารถพยากรณ์ข้อมูลใหม่ได้แม่นยำเพียงใด
# predict(X_test): ใช้โมเดลที่ฝึกไว้พยากรณ์ค่าเป้าหมาย (Profit) สำหรับข้อมูลในชุดทดสอบ (X_test)

print("y_test: ",y_test,"y_pred: ",y_pred) # ใช้โมเดลที่ฝึกเสร็จแล้วเพื่อพยากรณ์ค่าเป้าหมาย และเปรียบเทียบกับค่าจริง (y_test)

mse = mean_squared_error(y_test, y_pred) # ใช้วัดความคลาดเคลื่อนระหว่างค่าจริง (y_test) และค่าที่พยากรณ์ (y_pred)
rmse = math.sqrt(mse) # คำนวณค่า Root Mean Squared Error (RMSE) 

print("mse: ",mse, "rsme: ", rmse)

plt.scatter([1,2,3,4,5,6,7,8,9,10],y_test,color='red') # [1,2,3,4,5,6,7,8,9,10]: ใช้แทนตำแหน่งจุดในแกน X
plt.scatter([1,2,3,4,5,6,7,8,9,10],y_pred,color='blue')
plt.title("StartUp")
plt.xlabel("X Axis")
plt.ylabel("Profit")
plt.grid() # เพิ่มเส้นตารางเพื่อให้กราฟดูง่ายขึ้น
plt.show()
