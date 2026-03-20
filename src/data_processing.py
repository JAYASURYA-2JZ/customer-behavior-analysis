# ============================================================
# RAW DATA → RFM FEATURE ENGINEERING
# ============================================================

import pandas as pd

# ============================================================
# STEP 1: LOAD RAW DATASET
# ============================================================
# This should be your ORIGINAL dataset (transactions / visits)
raw_data = pd.read_csv("raw_ecommerce_sample.csv")

print("Raw data loaded")
print("Shape:", raw_data.shape)

# ============================================================
# STEP 2: BASIC PREPROCESSING
# ============================================================

# Convert visit_date to datetime
raw_data['visit_date'] = pd.to_datetime(raw_data['visit_date'])

# ============================================================
# STEP 3: DEFINE REFERENCE DATE
# ============================================================
# Usually the latest date in the dataset
reference_date = raw_data['visit_date'].max()

print("Reference date:", reference_date.date())

# ============================================================
# STEP 4: CALCULATE RFM FEATURES
# ============================================================

# -------- RECENCY --------
last_visit = raw_data.groupby('customer_id')['visit_date'].max()
recency_days = (reference_date - last_visit).dt.days

# -------- FREQUENCY --------
# Number of unique sessions / orders per customer
frequency = raw_data.groupby('customer_id')['session_id'].nunique()

# -------- MONETARY --------
# Total amount spent by customer
monetary = raw_data.groupby('customer_id')['total_amount'].sum()

# ============================================================
# STEP 5: COMBINE INTO RFM DATAFRAME
# ============================================================

rfm_data = pd.DataFrame({
    'customer_id': frequency.index,
    'frequency': frequency.values,
    'monetary': monetary.values,
    'recency_days': recency_days.values
})

print("\nRFM feature engineering completed")
print(rfm_data.head())

# ============================================================
# STEP 6: SAVE ENGINEERED DATA
# ============================================================

rfm_data.to_csv("rfm_engineered_data.csv", index=False)

print("\n✅ RFM engineered data saved as rfm_engineered_data.csv")
print("This file is ready for your FINAL inference code")