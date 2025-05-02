import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("medical_data_tfidf.csv")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(probability=True, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"دقة النموذج: {accuracy * 100:.2f}%")

print("\nتقرير التصنيف:")
print(classification_report(y_test, y_pred))

print("\nمصفوفة الارتباك:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, "medical_classifier_model.pkl")
print("تم حفظ النموذج في 'medical_classifier_model.pkl'")