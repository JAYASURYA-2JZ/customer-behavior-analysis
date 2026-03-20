import gradio as gr
import pandas as pd
import joblib

# ===============================
# LOAD MODELS
# ===============================
churn_model = joblib.load("churn_random_forest_model.pkl")
repeat_model = joblib.load("repeat_purchase_logistic_model.pkl")

engineered_data = None

# ===============================
# STATUS LOGIC
# ===============================
def get_status(churn_prob, repeat_prob, monetary):
    if churn_prob > 0.7:
        return "Churned"
    elif churn_prob > 0.4:
        return "At Risk"
    elif repeat_prob > 0.7:
        return "Loyal"
    elif monetary > 2000:
        return "High Value"
    else:
        return "Regular"

# ===============================
# RUN INFERENCE
# ===============================
def run_inference(df):

    df = df.copy()

    X_churn = df[['frequency','monetary']]
    X_rp = df[['recency_days','monetary']]

    df['churn_probability'] = churn_model.predict_proba(X_churn)[:,1]
    df['repeat_probability'] = repeat_model.predict_proba(X_rp)[:,1]

    df['status'] = df.apply(
        lambda r: get_status(r["churn_probability"],
                             r["repeat_probability"],
                             r["monetary"]), axis=1)

    return df

# ===============================
# SUMMARY CALCULATION
# ===============================
def calculate_summary(df):

    total = len(df)

    churn_pct = (df["churn_probability"] > 0.5).mean()*100
    repeat_pct = (df["repeat_probability"] > 0.5).mean()*100

    high_value_pct = (df["status"]=="High Value").mean()*100
    loyal_pct = (df["status"]=="Loyal").mean()*100
    risk_pct = (df["status"]=="At Risk").mean()*100
    churned_pct = (df["status"]=="Churned").mean()*100

    summary = f"""
Total Customers: {total}

Likely to Churn: {churn_pct:.1f}%
Likely to Repeat: {repeat_pct:.1f}%

High Value: {high_value_pct:.1f}%
Loyal: {loyal_pct:.1f}%
At Risk: {risk_pct:.1f}%
Churned: {churned_pct:.1f}%
"""
    return summary

# ===============================
# CSV UPLOAD
# ===============================
def upload_csv(file):

    global engineered_data
    engineered_data = pd.read_csv(file.name)

    result = run_inference(engineered_data)
    summary = calculate_summary(result)

    return result, summary

# ===============================
# SEARCH CUSTOMER
# ===============================
def search_customer(customer_id):

    if engineered_data is None:
        return "Upload CSV first","","",""

    customer = engineered_data[
        engineered_data["customer_id"] == int(customer_id)
    ]

    if customer.empty:
        return "Customer not found","","",""

    customer = run_inference(customer)
    row = customer.iloc[0]

    return (
        row["status"],
        f"{row['churn_probability']:.2f}",
        f"{row['repeat_probability']:.2f}",
        f"₹{row['monetary']}"
    )

# ===============================
# SAVE & APPEND
# ===============================
def save_results():

    global engineered_data

    if engineered_data is None:
        return "No data"

    df = run_inference(engineered_data)

    file = "prediction_results_log.csv"

    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)

    return "Saved & appended successfully"

# ===============================
# UI
# ===============================
with gr.Blocks(title="Customer Behavior Dashboard") as demo:

    gr.Markdown("# Customer Behavior Dashboard")

    with gr.Row():
        file_input = gr.File(label="Upload Engineered CSV")
        upload_btn = gr.Button("Run Predictions")

    summary_box = gr.Textbox(label="Customer Summary", lines=8)
    output_table = gr.Dataframe(label="Prediction Output")

    upload_btn.click(upload_csv, file_input, [output_table, summary_box])

    gr.Markdown("## Search Customer")

    cust_id = gr.Textbox(label="Enter Customer ID")
    search_btn = gr.Button("Search")

    status = gr.Textbox(label="Status")
    churn = gr.Textbox(label="Churn Probability")
    repeat = gr.Textbox(label="Repeat Probability")
    money = gr.Textbox(label="Monetary")

    search_btn.click(
        search_customer,
        cust_id,
        [status, churn, repeat, money]
    )

     
     
    save_btn = gr.Button("Save & Append Results")
    save_msg = gr.Textbox()

    save_btn.click(save_results, None, save_msg)

demo.launch()
