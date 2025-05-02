import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("medical_data_processed.csv")

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['cleaned_text']).toarray()

y = df['label']

feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(X, columns=feature_names)

tfidf_df['label'] = y

tfidf_df.to_csv("medical_data_tfidf.csv", index=False)

print("تم تحويل النصوص إلى أرقام باستخدام TF-IDF وحفظها في 'medical_data_tfidf.csv'")
print("أول 5 صفوف من البيانات المحولة:")
print(tfidf_df.head())