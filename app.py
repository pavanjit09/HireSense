from flask import Flask, render_template, request
import pickle
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

catboost_model = pickle.load(open("catboost_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_employee_attrition.pkl", "rb"))

@app.route('/charts')
def charts():
    data = pd.read_csv("HireSense_Job_Placement_Data.csv")  
    plt.figure(figsize=(6,4))
    sns.countplot(x='gender', hue='status', data=data.replace({'gender': {'M': 'Male', 'F': 'Female'}, 'status': {0: 'Not Placed', 1: 'Placed'}}))
    plt.title('Placement by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return render_template('charts.html', chart=image_base64)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/job')
def job_page():
    return render_template('predict_job.html')

@app.route('/attrition')
def attrition_page():
    return render_template('predict_attrition.html')

@app.route('/predict_job', methods=['POST'])
def predict_job():
    input_data = {
        'sl_no': 999, 
        'gender': str({'Male': 0, 'Female': 1}[request.form['gender']]),
        'ssc_p': float(request.form['ssc_p']),
        'hsc_p': float(request.form['hsc_p']),
        'degree_p': float(request.form['degree_p']),
        'workex': str({'No': 0, 'Yes': 1}[request.form['workex']]), 
        'etest_p': float(request.form['etest_p']),
        'specialisation': str({'Mkt&HR': 0, 'Mkt&Fin': 1}[request.form['specialisation']]), 
        'mba_p': float(request.form['mba_p']),
    }

    df = pd.DataFrame([input_data])
    prediction = catboost_model.predict(df)[0]
    result = "Placed" if prediction == 1 else "Not Placed"

    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(df)

    shap.initjs()
    force_html = shap.force_plot(explainer.expected_value, shap_values[0], df.iloc[0], show=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_html.html()}</body>"

    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    shap_bar_image = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    return render_template('result.html', prediction=result, shap_html=shap_html, shap_image=shap_bar_image)

@app.route('/predict_attrition', methods=['POST'])
def predict_attrition():
    data = {
        'satisfaction_level': float(request.form['satisfaction_level']),
        'last_evaluation': float(request.form['last_evaluation']),
        'number_project': int(request.form['number_project']),
        'average_montly_hours': int(request.form['average_montly_hours']),
        'time_spend_company': int(request.form['time_spend_company']),
        'work_accident': int(request.form['work_accident']),
        'promotion_last_5years': int(request.form['promotion_last_5years']),
        'department': request.form['department'],
        'salary': request.form['salary']
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    expected_columns = [
        'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
        'time_spend_company', 'work_accident', 'promotion_last_5years',
        'department_RandD', 'department_accounting', 'department_hr',
        'department_management', 'department_marketing', 'department_product_mng',
        'department_sales', 'department_support', 'department_technical',
        'salary_high', 'salary_low', 'salary_medium'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    prediction = xgb_model.predict(df)[0]
    result = "Employee will leave" if prediction == 1 else "Employee will stay"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
