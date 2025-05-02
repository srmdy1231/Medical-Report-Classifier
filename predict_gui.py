import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tkinter as tk
from tkinter import ttk
import csv

nltk.data.path.append('C:/nltk_data')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df = pd.read_csv("medical_data_processed.csv")
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(df['cleaned_text'])
clf = joblib.load("medical_classifier_model.pkl")

def predict_text():
    input_text = text_entry.get("1.0", tk.END).strip()
    if not input_text:
        result_label.config(text="الرجاء إدخال نص!", foreground="red")
        return
    
    cleaned_text = clean_text(input_text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    prediction = clf.predict(text_vector)[0]
    confidence = max(clf.predict_proba(text_vector)[0])
    result_label.config(text=f"التنبؤ: {prediction} (ثقة: {confidence:.2f})", foreground="green")

def save_prediction():
    input_text = text_entry.get("1.0", tk.END).strip()
    prediction_text = result_label.cget("text")
    if not prediction_text or "الرجاء" in prediction_text:
        result_label.config(text="الرجاء التنبؤ أولاً!", foreground="red")
        return
    with open("predictions_history.csv", "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([input_text, prediction_text])
    result_label.config(text=f"{prediction_text} (تم الحفظ)", foreground="blue")

root = tk.Tk()
root.title("تصنيف التقارير الطبية")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

title_label = ttk.Label(root, text="تصنيف التقارير الطبية", font=("Arial", 16, "bold"), background="#f0f0f0")
title_label.pack(pady=10)

ttk.Label(root, text="أدخل التقرير الطبي:", background="#f0f0f0").pack(pady=5)
text_entry = tk.Text(root, height=5, width=50, font=("Arial", 10))
text_entry.pack(pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)
ttk.Button(button_frame, text="تنبؤ", command=predict_text).pack(side="left", padx=5)
ttk.Button(button_frame, text="حفظ التنبؤ", command=save_prediction).pack(side="left", padx=5)

result_label = ttk.Label(root, text="", font=("Arial", 12), background="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()