import pandas as pd

df = pd.read_csv("Healthcare Documentation Database.csv")

df['label'] = df.apply(lambda row: 'healthy' if 'normal' in str(row['description']).lower() or 'normal' in str(row['keywords']).lower() else 
                           'diabetes' if 'diabetes' in str(row['description']).lower() or 'diabetes' in str(row['keywords']).lower() or 'endocrinology' in str(row['medical_specialty']).lower() else 
                           'cancer' if 'cancer' in str(row['description']).lower() or 'oncology' in str(row['keywords']).lower() or 'oncology' in str(row['medical_specialty']).lower() else 
                           'infection' if 'infection' in str(row['description']).lower() or 'infectious' in str(row['keywords']).lower() or 'infectious' in str(row['medical_specialty']).lower() else 'unknown', axis=1)

df = df[df['label'] != 'unknown']

df.rename(columns={'transcription': 'text'}, inplace=True)

df[['text', 'label']].to_csv("medical_data_initial.csv", index=False)

print("تم إنشاء ملف 'medical_data_initial.csv' بنجاح!")
print(df[['text', 'label']].head())