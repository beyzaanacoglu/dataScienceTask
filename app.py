# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("Titanic Survival Prediction (Beginner Style with Visualizations)")

# ------------------------------------------------------------------
# Load dataset (online by default, fallback to local file)
# ------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    """Read the Titanic CSV from the web; if it fails, try a local copy."""
    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        df = pd.read_csv("titanic.csv")  # expects a local file in the same folder
    return df

data = load_data()

# ------------------------------------------------------------------
# Show raw data preview
# ------------------------------------------------------------------
st.subheader("Dataset preview (first 5 rows)")
st.write(data.head())

# ------------------------------------------------------------------
# Statistical summary
# ------------------------------------------------------------------
st.subheader("Statistical summary")
st.write(data.describe(include="all"))

# Helper to render seaborn/matplotlib figures in Streamlit
def st_seaborn(fig):
    st.pyplot(fig)

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------

# 1. Survival count by sex
st.subheader("Survival by Sex")
fig1, ax1 = plt.subplots()
sns.countplot(data=data, x="sex", hue="survived", ax=ax1)
ax1.set_xlabel("Sex")
ax1.set_ylabel("Count")
st_seaborn(fig1)

# 2. Survival count by passenger class
st.subheader("Survival by Passenger Class")
fig2, ax2 = plt.subplots()
sns.countplot(
    data=data,
    x="class",
    hue="survived",
    order=["First", "Second", "Third"],
    ax=ax2,
)
ax2.set_xlabel("Class")
ax2.set_ylabel("Count")
st_seaborn(fig2)

# 3. Age distribution coloured by survival
st.subheader("Age distribution by Survival")
fig3, ax3 = plt.subplots()
sns.histplot(
    data=data,
    x="age",
    hue="survived",
    kde=True,
    bins=30,
    ax=ax3,
)
ax3.set_xlabel("Age")
st_seaborn(fig3)

# 4. Fare distribution (boxplot) by class
st.subheader("Fare distribution by Class")
fig4, ax4 = plt.subplots()
sns.boxplot(
    data=data,
    x="class",
    y="fare",
    order=["First", "Second", "Third"],
    ax=ax4,
)
ax4.set_xlabel("Class")
ax4.set_ylabel("Fare")
st_seaborn(fig4)

# 5. Correlation heatmap of numeric features
st.subheader("Correlation heatmap of numeric features")
numeric_cols = data.select_dtypes(include="number")
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax5)
st_seaborn(fig5)

# ------------------------------------------------------------------
# Machine‑learning model (RandomForest)
# ------------------------------------------------------------------
# Feature engineering – simple encoding for sex & embarkation point
X = data[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
X["sex"] = X["sex"].map({"male": 0, "female": 1})
X["embarked"] = X["embarked"].fillna("S")
X = pd.get_dummies(X, columns=["embarked"], drop_first=True)
X = X.fillna(0)

y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# Prediction UI
# ------------------------------------------------------------------
st.subheader("Make a prediction")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input(
        "Siblings/Spouses aboard", min_value=0, max_value=10, value=0
    )
    parch = st.number_input(
        "Parents/Children aboard", min_value=0, max_value=10, value=0
    )
    fare = st.number_input("Ticket fare", min_value=0.0, value=32.0)
    embarked = st.selectbox(
        "Port of embarkation", ["C", "Q", "S"], index=2
    )  # default to 'S'

with col2:
    st.write("\n")  # spacer

# Build a dataframe that matches the training feature set
input_df = pd.DataFrame(
    {
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked],
    }
)
input_df["sex"] = input_df["sex"].map({"male": 0, "female": 1})
input_df = pd.get_dummies(input_df, columns=["embarked"], drop_first=True)

# Ensure all columns expected by the model are present
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("Survived! 🎉")
    else:
        st.error("Did not survive... 😢")

st.caption(
    "Data source: https://github.com/mwaskom/seaborn-data/tree/master/titanic.csv"
)
