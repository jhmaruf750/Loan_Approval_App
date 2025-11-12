import streamlit as st
import pandas as pd
import numpy as np

import pickle as pk

# Configure page
st.set_page_config(page_title="Loan Prediction App", page_icon="💰", layout="wide")

# Load models
model=pk.load(open('model.pkl', 'rb'))
scalar=pk.load(open('scaler.pkl', 'rb'))

# Header with styling
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>💰 Loan Prediction App</h1>
        <p style='font-size: 18px; color: #666;'>Check your loan eligibility instantly</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Information")
    no_of_dep = st.slider("👨‍👩‍👧‍👦 No of Dependents", 0, 5, 2)
    grad = st.selectbox("🎓 Education Level", ["Not Graduate", "Graduate"])
    self_emp = st.selectbox("💼 Self Employed", ["No", "Yes"])

with col2:
    st.subheader("💵 Financial Information")
    Annual_income = st.slider("💰 Annual Income (BDT)", 0, 10000000, 9600000, step=100000, format="৳%d")
    Loan_ammount = st.slider("📊 Loan Amount (BDT)", 0, 5000000, 29900000, step=100000, format="৳%d")
    Loan_duration = st.slider("⏰ Loan Term (in months)", 0, 480, 12)

# Credit Information
st.subheader("📈 Credit Information")
col3, col4 = st.columns(2)

with col3:
    cibil = st.slider("⭐ CIBIL Score", 300, 900, 778)

with col4:
    assets = st.slider("🏠 Assets (BDT)", 0, 100000000, 50700000, step=100000, format="৳%d")   

st.markdown("---")

# Convert categorical inputs to numeric
if grad == 'Graduate':
    grad_s = 1
else:
    grad_s = 0
    
if self_emp == 'Yes':
    self_emp_s = 1
else:
    self_emp_s = 0

# Create prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    if st.button("🔍 Predict Loan Eligibility", use_container_width=True):
        input_data = pd.DataFrame(
            [[no_of_dep, grad_s, self_emp_s, Annual_income, Loan_ammount, Loan_duration, cibil, assets]],
            columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
        )
        scaled_data = scalar.transform(input_data)
        prediction = model.predict(scaled_data)
        
        st.markdown("---")
        
        # Display results with styling
        if prediction[0] == 1:
            st.success("✅ **Congratulations! You are eligible for the loan.**")
            st.balloons()
            
            # Show summary
            with st.expander("📊 Your Application Summary", expanded=True):
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write(f"**Dependents:** {no_of_dep}")
                    st.write(f"**Education:** {grad}")
                    st.write(f"**Self Employed:** {self_emp}")
                with col_res2:
                    st.write(f"**Annual Income:** ৳{Annual_income:,}")
                    st.write(f"**Loan Amount:** ৳{Loan_ammount:,}")
                    st.write(f"**CIBIL Score:** {cibil}")
        else:
            st.error("❌ **Sorry, you are not eligible for the loan at this time.**")
            st.info("💡 Consider improving your CIBIL score or financial profile and try again.")
            
            # Show summary
            with st.expander("📊 Your Application Summary", expanded=True):
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write(f"**Dependents:** {no_of_dep}")
                    st.write(f"**Education:** {grad}")
                    st.write(f"**Self Employed:** {self_emp}")
                with col_res2:
                    st.write(f"**Annual Income:** ৳{Annual_income:,}")
                    st.write(f"**Loan Amount:** ৳{Loan_ammount:,}")
                    st.write(f"**CIBIL Score:** {cibil}")            

pred_data = pd.DataFrame(
    [[2, 1, 0, 9600000, 29900000, 12, 778, 50700000]],
    columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
)

# Beautiful footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 40px;'>
        <p style='font-size: 16px; color: white; font-weight: bold; margin: 0;'>
            Created by: Md. Zahid Hasan Maruf
        </p>
        <p style='font-size: 14px; color: #e0e0e0; margin: 5px 0 0 0;'>
            RUET CSE'21 | Bangladesh
        </p>
        <p style='font-size: 12px; color: #b0b0b0; margin-top: 10px;'>
            🏦 Machine Learning Based Loan Prediction System 
        </p>
    </div>
    """, unsafe_allow_html=True)

