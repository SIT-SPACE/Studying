import seaborn as sns
import pandas as pd #Pandas: ใช้จัดการข้อมูลในรูปแบบตาราง (DataFrame)
import matplotlib.pyplot as plt #Matplotlib: ใช้สร้างกราฟเพื่อวิเคราะห์และแสดงผลข้อมูล

data = pd.read_csv('D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/iris.csv') #โหลดข้อมูล
df = pd.DataFrame(data) #แปลงข้อมูลเป็น DataFrame โดยใช้ไลบารี่ pandas

sns.boxplot(x=df['species'], y=df['sepal_length'])
plt.show()

box_plot_data = [df['sepal_length'], 
                 df['sepal_width'], #ตัวแปร box_plot_data เป็น List (ลิสต์) ที่เก็บ Pandas Series (คอลัมน์ใน DataFrame) หลายตัว
                 df['petal_length'], # Pandas Series คือโครงสร้างข้อมูลที่มีลักษณะคล้ายกับ 1D array และมีดัชนี (Index) ที่ติดตามค่าในคอลัมน์
                 df['petal_width']] # box_plot_data เป็นลิสต์ของหลายๆ Series (หลายคอลัมน์), การใช้งานนี้เหมาะสมกับการสร้างกราฟหลายชุดข้อมูลในกราฟเดียว
plt.boxplot(box_plot_data,patch_artist=True,
            labels=['sepal_length', 'sepal_width',
                   'petal_length', 'petal_width'])
plt.show()

