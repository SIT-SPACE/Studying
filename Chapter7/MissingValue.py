import numpy as np
import scipy as sp

filename = 'D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/data.tsv'
data = np.genfromtxt(filename, delimiter="\t") # np.genfromtxt อ่านข้อมูลจากไฟล์ / ตัวแบ่งคอลัมน์ (delimiter) 
#ค่าต่างๆ เช่น 1 2272: แปลงเป็น [1.000e+00 2.272e+03] แสดงในรูปแบบเลขยกกำลัง (Scientific Notation)
print(data)

print(data[np.isnan(data)])

x,y = data[:,0], data[:,1] # : หมายถึงการเลือกทุกแถว (ทั้งหมด) / เลือกข้อมูลในคอลัมน์ 0,1 (คอลัมน์ที่ 0,1) ทั้งหมด
print("X:",x, "Y:",y)

x = x[~np.isnan(y)] # เลือกเฉพาะค่าของ x ที่ตำแหน่งของ y ไม่เป็น NaN
y = y[~np.isnan(y)] # เลือกเฉพาะค่าที่ True (ในที่นี้คือค่าที่ไม่เป็น NaN)

print("X:",x, "Y:",y)

