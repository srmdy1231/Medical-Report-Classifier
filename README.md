# Medical Report Classifier

## العربية
مشروع لتصنيف التقارير الطبية إلى 4 فئات (healthy, diabetes, cancer, infection) باستخدام معالجة النصوص الطبيعية (NLP) والتعلم الآلي.

### الخطوات
1. قراءة البيانات: `generate_medical_data.py`
2. تنظيف النصوص: `process_medical_text.py`
3. تحويل النصوص إلى أرقام: `convert_to_tfidf.py`
4. تدريب النموذج: `train_classifier.py`
5. اختبار النموذج: `predict_new_data.py`
6. واجهة المستخدم: `predict_gui.py`

### البيانات
- البيانات المستخدمة: "Healthcare Documentation Database" من Kaggle.
- رابط البيانات: [Healthcare Documentation Database] (https://www.kaggle.com/datasets/harshitstark/healthcare-documentation-database) 

### طريقة التشغيل
1. فعّلي البيئة الافتراضية:
    2. ثبتي المكتبات:pip install pandas nltk scikit-learn
 3. شغلي الملفات بالترتيب:
 4. 
### الميزات
- تصنيف التقارير الطبية بدقة عالية (مثال: تصنيف حالة صحية بنسبة ثقة 0.87).
- واجهة مستخدم سهلة باستخدام `tkinter`.
- إظهار نسبة الثقة مع كل تنبؤ.
- إمكانية حفظ التنبؤات في ملف CSV.

## English
A project to classify medical reports into 4 categories (healthy, diabetes, cancer, infection) using Natural Language Processing (NLP) and Machine Learning.

### Steps
1. Load data: `generate_medical_data.py`
2. Clean text: `process_medical_text.py`
3. Convert text to numbers: `convert_to_tfidf.py`
4. Train the model: `train_classifier.py`
5. Test the model: `predict_new_data.py`
6. User interface: `predict_gui.py`

### Data
- Data used: "Healthcare Documentation Database" from Kaggle.
- Data link: [Healthcare Documentation Database](https://www.kaggle.com/datasets/harshitstark/healthcare-documentation-database) 

### How to Run
1. Activate the virtual environment:
2. 2. Install the libraries:
   3. 
### Features
- High-accuracy medical report classification (e.g., classified a healthy case with 0.87 confidence).
- User-friendly interface using `tkinter`.
- Displays confidence score with each prediction.
- Ability to save predictions to a CSV file.
