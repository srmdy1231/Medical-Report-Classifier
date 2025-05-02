import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.data.path.append('C:/nltk_data')

df = pd.read_csv("medical_data_initial.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

print("البيانات بعد التنظيف:")
print(df[['text', 'cleaned_text', 'label']].head())

df.to_csv("medical_data_processed.csv", index=False)
print("تم حفظ البيانات المنظفة في 'medical_data_processed.csv'")