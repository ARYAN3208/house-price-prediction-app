import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("house_price_model.pkl", "rb"))

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction System")
st.markdown("Predict house prices using Machine Learning")

st.divider()

# ---------------- NUMERICAL INPUTS ----------------
st.sidebar.header("Numerical Features")

MSSubClass = st.sidebar.number_input("MSSubClass", 20, 200, 60)
LotArea = st.sidebar.number_input("Lot Area (sqft)", 1000, 20000, 8000)
OverallCond = st.sidebar.slider("Overall Condition (1–10)", 1, 10, 5)
YearBuilt = st.sidebar.slider("Year Built", 1870, 2025, 2000)
YearRemodAdd = st.sidebar.slider("Year Remodeled", 1870, 2025, 2005)
BsmtFinSF2 = st.sidebar.number_input("Basement Finished Area 2", 0, 5000, 0)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area", 0, 6000, 800)

# ---------------- CATEGORICAL INPUTS ----------------
st.sidebar.header("Categorical Features")

MSZoning = st.sidebar.selectbox(
    "MS Zoning",
    ["0", "C (all)", "FV", "RH", "RL", "RM"]
)

LotConfig = st.sidebar.selectbox(
    "Lot Configuration",
    ["Corner", "CulDSac", "FR2", "FR3", "Inside"]
)

BldgType = st.sidebar.selectbox(
    "Building Type",
    ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"]
)

Exterior1st = st.sidebar.selectbox(
    "Exterior 1st",
    ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock",
     "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Plywood",
     "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"]
)

# ---------------- ONE HOT ENCODING ----------------
def encode_choice(choice, options):
    return [1 if choice == opt else 0 for opt in options]

MSZoning_cols = ["0","C (all)","FV","RH","RL","RM"]
LotConfig_cols = ["Corner","CulDSac","FR2","FR3","Inside"]
BldgType_cols = ["1Fam","2fmCon","Duplex","Twnhs","TwnhsE"]
Exterior_cols = ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock",
                 "CemntBd","HdBoard","ImStucc","MetalSd","Plywood",
                 "Stone","Stucco","VinylSd","Wd Sdng","WdShing"]

MSZoning_encoded = encode_choice(MSZoning, MSZoning_cols)
LotConfig_encoded = encode_choice(LotConfig, LotConfig_cols)
BldgType_encoded = encode_choice(BldgType, BldgType_cols)
Exterior_encoded = encode_choice(Exterior1st, Exterior_cols)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict House Price"):
    input_data = np.array([[
        MSSubClass, LotArea, OverallCond, YearBuilt, YearRemodAdd,
        BsmtFinSF2, TotalBsmtSF,
        *MSZoning_encoded,
        *LotConfig_encoded,
        *BldgType_encoded,
        *Exterior_encoded
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted House Price: ₹ {prediction:,.2f}")
    st.balloons()
