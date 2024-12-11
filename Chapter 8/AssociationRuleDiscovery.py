import pandas as pd
from apyori import apriori # ไลบรารีสำหรับการวิเคราะห์ Association Rules โดยใช้ Apriori Algorithm

df_data = pd.read_csv('D:/Users/phisi/OneDrive - Thammasat University/Coding/Studying/Data/retail_dataset.csv')

records = []
for i in range(0,315): # for i in range(0, 315): ลูปวนผ่านทุกแถว (ธุรกรรม) ในข้อมูล
    records.append([str(df_data.values[i,j]) # records.append([...]): เก็บรายการสินค้าทั้งหมดในแต่ละธุรกรรม (แถว) เป็นลิสต์
    for j in range(0,7) if str(df_data.values[i,j]) != 'nan']) #for j in range(0, 7): ลูปผ่านทุกคอลัมน์ / str(df_data.values[i, j]) != 'nan': กรองค่าที่ไม่ใช่ NaN (ค่าที่ว่าง)

results = list(apriori(records, min_support=0.1, min_confidence=0.51))
'''
Support: ความถี่หรือสัดส่วนที่รายการหรือกลุ่มรายการปรากฏในข้อมูลธุรกรรมทั้งหมด
Confidence: ความน่าจะเป็นที่รายการในฝั่งขวา (Consequent) จะปรากฏเมื่อมีรายการในฝั่งซ้าย (Antecedent)
'''
for i in range(len(results)):
    LHS = list(results[i][2][0][0]) #list(results[i][2][0][1]): แปลง frozenset ซึ่งเป็นชนิดข้อมูลที่ไม่สามารถแก้ไขได้ (Immutable) ให้เป็น list:
    RHS = list(results[i][2][0][1])
    support = results[i][1]
    confidence = results[i][2][0][2]
    lift = results[i][2][0][3]
    print("LHS: ",LHS,"=>","RHS: ", RHS)
    print("Support: ", support)
    print("Confidence: ",confidence)
    print("Lift: ",lift)
    print(10*"----")


''' # Output results
RelationRecord(
    items=frozenset({'Milk', 'Bread'}),
    support=0.12,
    ordered_statistics=[ # results[i][2]: ดึง ordered_statistics ซึ่งเป็นรายการย่อย (List) ที่เก็บข้อมูลเกี่ยวกับกฎ
        OrderedStatistic( # results[i][2][0]: เลือกรายการแรกของ ordered_statistics
            items_base=frozenset({'Milk'}),
            items_add=frozenset({'Bread'}), #results[i][2][0][1]: ดึง items_add จาก OrderedStatistic
            confidence=0.6,
            lift=1.2
        )
    ]
)
'''