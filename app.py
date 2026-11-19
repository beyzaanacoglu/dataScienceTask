Conversation Log
bu task için gerekli dosyaları tamamla. profesyonel olmasın aksne daha yeni kodlamaöğrenen birisi yapmış gibi görünsün
Oluşturulan dosyalar
Dosya	Açıklama	Bağlantı
app.py	Yeni başlayan birinin stilinde, basit bir Streamlit Titanic tahmin uygulaması	app.py
requirements.txt	Çalıştırmak için gereken paketler	requirements.txt
README.md	Kullanım talimatları ve proje açıklaması	README.md
Sonraki adımlar

Çalışma alanı olarak bu klasörü ayarlayın – C:\Users\beyza.anacoglu\.gemini\antigravity\scratch\titanic_streamlit dizinini aktif workspace olarak seçmek, ileride agy komutlarıyla (ör. agy run) kolayca yönetmenizi sağlar.

Python ortamı oluşturun (python -m venv venv ve venv\Scripts\activate).

Bağımlılıkları yükleyin:

powershell
pip install -r requirements.txt

Titanic veri setini (titanic.csv) aynı klasöre koyun.

Uygulamayı başlatın:

powershell
streamlit run app.py

Kod, yeni başlayan birinin sıkça yaptığı “temiz olmayan” düzenlemeler ve yorumlar içermekte; profesyonel bir yapıdan ziyade öğrenme sürecinde yazılmış gibi görünüyor.

Herhangi bir dosyada değişiklik isterseniz veya ek bir şey eklemek isterseniz, bana bildirin!

AppRequirements
README.md kısa olsun ve profesyonel şekilde yapıldığını anlatsın

Güncellenmiş README (kısa ve profesyonel)

README.md

Bu dosyada, projenin ne olduğu, nasıl çalıştırılacağı ve temel gereksinimler net bir şekilde özetlenmiştir. Başka bir değişiklik isterseniz, bana bildirin!

tüm dosyalardaki yorum satırlarını sil. her şey ingilizce olmalı

All files have been updated:

app.py – All Turkish comments removed, code now fully in English.
app.py

README.md – Concise, professional description in English.
README.md

requirements.txt – Unchanged (already minimal).
requirements.txt

The project is now fully English and comment‑free, ready for a clean run. Let me know if you need any further tweaks!

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.title("Titanic Survival Prediction (Newbie Version)")
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    return df
data = load_data()
st.subheader('First 5 rows of the dataset')
st.write(data.head())
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X = X.fillna(0)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
st.subheader('Enter passenger information for prediction')
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox('Ticket class', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
with col2:
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    sibsp = st.number_input('Siblings/Spouses aboard', min_value=0, max_value=10, value=0)
    parch = st.number_input('Parents/Children aboard', min_value=0, max_value=10, value=0)
    fare = st.number_input('Ticket fare', min_value=0.0, value=32.0)
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare]
})
input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
if st.button('Predict'):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success('Survived! 🎉')
    else:
        st.error('Did not survive... 😢')