import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42) # rastgele sayıların tekrarlanabilirliğini sağlamak için sabit bir değer belirliyoruz.

n_samples = 2000

df = pd.DataFrame({
    'age': np.random.randint(18, 66, n_samples),
    'income': np.random.randint(3000, 30001, n_samples),
    'debt': np.random.randint(0, 50001, n_samples),
    'credit_score': np.random.randint(300, 901, n_samples),
    'employment_years': np.random.randint(0, 41, n_samples)
})

def check_approval(row):
    if (row["income"] > 8000 and 
        row["credit_score"] > 600 and 
        row["debt"] < 20000 and 
        row["employment_years"] > 2):
        return 1
    else:
        return 0

df["approved"] = df.apply(check_approval, axis=1)

# %10 oranında rastgele gürültü ekliyoruz.
noise = np.random.rand(len(df)) < 0.1  #%10 oranında rastgele gürültü ekliyoruz.

df.loc[noise, "approved"] = 1 - df.loc[noise, "approved"]  # 1→0, 0→1



X = df[['age', 'income', 'debt', 'credit_score', 'employment_years']] #bağımsız değişkenlerimizdir. yani özelliklerimizdir.
y = df['approved'] #bağımlı değişkenimizdir. yani hedef değişkenimizdir.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #verilerimizin %20 sini test verisi ½80 sini eğitim verisi olarak seçer.

model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=100 ağaç sayısını belirler. Ağaç sayısı arttıkça modelin doğruluğu artar denilebilir ama bu veriye göre değişir.
model.fit(X_train, y_train) #modelimizi eğitiyoruz.

y_prediction = model.predict(X_test) #modelimizi test ediyoruz. tahmin yapmamızı sağlar.

print("Accuacy:", accuracy_score(y_test, y_prediction)) #doğruluk oranını hesaplar. Modelimizin ne kadar doğru tahmin yaptığını gösterir.

cm = confusion_matrix(y_test, y_prediction) #karışıklık matrisini hesaplar. Modelimizin ne kadar doğru tahmin yaptığını gösterir.
print("Confusion Matrix:\n", cm) #karışıklık matrisini yazdırır.
print("Classification Report:\n", classification_report(y_test, y_prediction)) #sınıflandırma raporunu yazdırır. Modelimizin ne kadar doğru tahmin yaptığını gösterir.
print("Feature Importances:\n", model.feature_importances_) #özelliklerin önem derecelerini yazdırır. Modelimizin hangi özelliklerin ne kadar önemli olduğunu gösterir.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()
# Özelliklerin önem derecelerini görselleştirmek için bir grafik çiziyoruz.
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8,6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Önemi")
plt.title("Özellik Önem Grafiği (Feature Importance)")
plt.tight_layout()
plt.show()
