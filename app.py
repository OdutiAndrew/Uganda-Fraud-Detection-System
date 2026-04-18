import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Uganda Fraud Detector", page_icon="S", layout="centered")

@st.cache_resource
def load_artifacts():
    model    = joblib.load('model_artifacts/xgboost_fraud_model.pkl')
    scaler   = joblib.load('model_artifacts/scaler.pkl')
    features = joblib.load('model_artifacts/feature_names.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()
NUM_COLS = ['amount_ugx','sender_age','account_age_days','prev_txn_count_30d','risk_score']

st.title("Uganda Financial Fraud Detection System")
st.markdown("*SACCO and Mobile Money Transaction Risk Assessment*")
st.divider()

col1, col2 = st.columns(2)
with col1:
    txn_type = st.selectbox("Transaction Type",
        ['Mobile Money Transfer','SACCO Withdrawal','SACCO Deposit',
         'Mobile Money Deposit','Loan Repayment','Bill Payment'])
    network  = st.selectbox("Network Provider", ['MTN MoMo','Airtel Money','SACCO Account'])
    district = st.selectbox("District",
        ['Kampala','Wakiso','Mukono','Gulu','Mbarara','Jinja',
         'Mbale','Lira','Arua','Fort Portal'])
    amount   = st.number_input("Transaction Amount (UGX)", 1000, 5_000_000, 200_000, 10_000)
    hour     = st.slider("Hour of Day", 0, 23, 14)

with col2:
    sender_age       = st.number_input("Sender Age", 18, 80, 35)
    account_age_days = st.number_input("Account Age (days)", 1, 3650, 365)
    prev_txn_30d     = st.number_input("Transactions last 30 days", 0, 100, 5)
    failed_pin       = st.selectbox("Failed PIN Attempts", [0,1,2,3])
    is_new_recip     = st.checkbox("New Recipient?")
    same_net         = st.checkbox("Same Network Transfer?")
    device_chg       = st.checkbox("Device Change Detected?")
    is_weekend_chk   = st.checkbox("Weekend Transaction?")

if st.button("Analyse Transaction", type="primary", use_container_width=True):
    is_night     = 1 if (hour >= 22 or hour <= 5) else 0
    amount_tier  = min(4, int(np.digitize(amount, [50_000,200_000,500_000,1_000_000])))
    nalt         = 1 if (account_age_days < 30 and amount > 500_000) else 0
    risk_score   = (failed_pin*0.3 + int(is_new_recip)*0.2 +
                    int(device_chg)*0.2 + is_night*0.1 + nalt*0.2)

    row = {f: 0 for f in FEATURES}
    row.update({'hour_of_day': hour, 'amount_ugx': amount, 'sender_age': sender_age,
                'account_age_days': account_age_days, 'prev_txn_count_30d': prev_txn_30d,
                'failed_pin_attempts': failed_pin, 'is_new_recipient': int(is_new_recip),
                'same_network': int(same_net), 'device_change': int(device_chg),
                'is_weekend': int(is_weekend_chk), 'is_night': is_night,
                'amount_tier': amount_tier, 'new_account_large_txn': nalt,
                'risk_score': risk_score})
    for col_name, val_name in [('transaction_type', txn_type),
                                ('network_provider', network),
                                ('district', district)]:
        key = f'{col_name}_{val_name}'
        if key in FEATURES: row[key] = 1

    X_in = pd.DataFrame([row])[FEATURES]
    num_present = [c for c in NUM_COLS if c in X_in.columns]
    X_in[num_present] = scaler.transform(X_in[num_present])

    proba = model.predict_proba(X_in)[0][1]
    pred  = int(proba >= 0.5)

    st.divider()
    if pred == 1:
        st.error(f"FRAUD ALERT — Risk Score: {proba*100:.1f}%")
        st.markdown("This transaction is flagged as **HIGH RISK**. Recommend blocking for review.")
    else:
        st.success(f"LEGITIMATE — Risk Score: {proba*100:.1f}%")
        st.markdown("Transaction appears **LOW RISK** and is approved for processing.")
    st.progress(float(proba))

    with st.expander("Risk Factor Breakdown"):
        factors = {
            "High Amount (>1M UGX)":      amount > 1_000_000,
            "Multiple Failed PINs":        failed_pin >= 2,
            "New Recipient":               is_new_recip,
            "Device Change":               device_chg,
            "Night Transaction (22-05)":   is_night == 1,
            "New Account + Large Amount":  nalt == 1,
        }
        for factor, triggered in factors.items():
            icon = "RED" if triggered else "OK"
            st.write(f"[{icon}] {factor}: {'TRIGGERED' if triggered else 'CLEAR'}")

st.divider()
st.caption("CSC8204 Project | Uganda Christian University | Easter 2026")
