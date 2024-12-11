from ydata_profiling import ProfileReport
from sklearn import datasets
import pandas as pd
import webbrowser

data = pd.read_csv('D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/iris.csv')#โหลด Iris Dataset ซึ่งเป็นชุดข้อมูลที่ประกอบด้วยข้อมูลเกี่ยวกับดอกไม้ประเภท Iris  มีคุณสมบัติ 4 ตัว เช่น sepal_length มีข้อมูลทั้งหมดเลย
df = pd.DataFrame(data)
profile = ProfileReport(df, title="Pandas Profiling Report", html={'style':{'full_width':True}})
profile.to_file("iris_profile_report.html") # บันทึกเป็นไฟล์ HTML
webbrowser.open("iris_profile_report.html") # เปิดไฟล์ HTML ในเว็บเบราว์เซอร์