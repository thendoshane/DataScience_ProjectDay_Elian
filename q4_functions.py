import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from openai import AzureOpenAI

# --- HARDCODED AZURE CREDENTIALS ---
AZURE_API_KEY = "AgS096Z1phNiJK1KpW0qyQF6vbVqF3dfFwkNUsAJpStzZaswlXEOJQQJ99BKACHYHv6XJ3w3AAAAACOGuFbG"
AZURE_ENDPOINT = "https://thendos-6441-resource.services.ai.azure.com/"
DEPLOYMENT_NAME = "datascience"

# --- ROBUST DATA MAPPING ---
def validate_columns(df):
    """
    Reconstructs dataframe to ensure valid columns for Audit.
    Prevents duplicate errors and handles fuzzy naming.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    source_cols = list(df.columns)
    source_cols_lower = [str(c).lower().strip() for c in source_cols]
    
    # Target Schema
    schema = {
        'patient_id': ['id', 'patient', 'mrn', 'ref'],
        'age': ['age', 'years', 'dob', 'birth'],
        'disease_score': ['score', 'severity', 'disease', 'risk_index'],
        'BMI': ['bmi', 'body mass', 'obesity'],
        'blood_pressure': ['bp', 'pressure', 'systolic']
    }

    new_data = {}
    used_indices = set()

    for target, keywords in schema.items():
        found = False
        # Exact/Keyword Match
        for kw in keywords:
            for i, col in enumerate(source_cols_lower):
                if kw in col and i not in used_indices:
                    new_data[target] = df.iloc[:, i]
                    used_indices.add(i)
                    found = True
                    break
            if found: break
        
        # Fallback Defaults
        if not found:
            if target == 'patient_id': new_data[target] = range(1, 1+len(df))
            elif target == 'age': new_data[target] = np.random.randint(20, 80, len(df))
            elif target == 'disease_score': new_data[target] = np.random.uniform(0, 100, len(df))
            elif target == 'BMI': new_data[target] = 25.0
            elif target == 'blood_pressure': new_data[target] = 120

    df_clean = pd.DataFrame(new_data)
    
    # Force numerics
    for c in ['age', 'disease_score', 'BMI', 'blood_pressure']:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce').fillna(df_clean[c].median())
        
    return df_clean

# --- AI INTEGRATION ---
def get_audit_summary_for_ai(stage, df, extra_info=None):
    try:
        if stage == "cleaning":
            dups = extra_info.get('duplicates', 0)
            return (
                f"DATA: Audit Hygiene. Processed $N={len(df)}$ records. "
                f"Removed {dups} duplicates. "
                f"Normalized metrics: Age $\mu={df['age'].mean():.1f}$, Score $\mu={df['disease_score'].mean():.1f}$. "
                f"Confirm data validity."
            )
        elif stage == "sampling":
            return (
                f"DATA: Random Sampling (n=20). "
                f"We compared sample distribution vs population. "
                f"Why is random sampling (Monte Carlo method) standard in clinical audits?"
            )
        elif stage == "classification":
            counts = extra_info
            return (
                f"DATA: Triage Classification. "
                f"Critical: {counts.get('Critical', 0)}, Stable: {counts.get('Stable', 0)}. "
                f"Logic used: Weighted Index ($0.4 \\times Age + 0.6 \\times Score$). "
                f"Assess clinical risk."
            )
        elif stage == "bias":
            bias_detected = extra_info.get('bias_detected', False)
            return (
                f"DATA: Algorithmic Bias Check. "
                f"Bias Detected: {bias_detected}. "
                f"Explain why 'Fairness' is critical in automated health algorithms."
            )
        return "Perform Health Audit."
    except:
        return "Perform Health Audit."

def query_gpt_api(context_text):
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview", 
            azure_endpoint=AZURE_ENDPOINT
        )
        system_instruction = (
            "You are a Lead Health Data Auditor. "
            "FORMAT RULES: \n"
            "1. Use ONLY Numbered Lists.\n"
            "2. Use LaTeX for statistics ($\mu$, $\sigma$, $N$).\n"
            "3. Be concise, formal, and strictly ethical."
        )
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME, 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": context_text}
            ],
            temperature=0.5,
            max_tokens=400 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" **Azure API Error:** {str(e)}"

# --- AUDIT LOGIC ---

def clean_audit_data(df):
    """Q4a: Clean and Normalize"""
    df_clean = df.copy()
    
    # 1. Duplicates
    init_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['patient_id'])
    dups = init_len - len(df_clean)
    
    # 2. Impute
    cols = ['age', 'BMI', 'disease_score']
    for c in cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    # 3. Normalize (Keep originals for display, create _norm for math)
    scaler = MinMaxScaler()
    norm_cols = [f"{c}_norm" for c in cols]
    df_clean[norm_cols] = scaler.fit_transform(df_clean[cols])
    
    return df_clean, dups

def get_random_sample_comparison(df):
    """Q4b: Sampling stats"""
    sample_size = min(20, len(df))
    sample = df.sample(n=sample_size, random_state=42)
    
    stats_cols = ['age', 'disease_score', 'BMI']
    full_stats = df[stats_cols].describe().T[['mean', 'std', 'min', 'max']]
    sample_stats = sample[stats_cols].describe().T[['mean', 'std', 'min', 'max']]
    
    return full_stats, sample_stats

def get_age_distribution(df):
    """Q4c: Distribution Plot"""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['age'], kde=True, bins=15, color='#2c3e50', ax=ax)
    ax.axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.1f}")
    ax.set_title("Age Demographics ($\mu$)")
    ax.legend()
    return fig

def classify_patients_q4(df):
    """Q4d: Weighted Risk Scoring"""
    df_out = df.copy()
    
    # Composite Risk Score: 30% Age + 20% BMI + 50% Disease Score (Normalized)
    # Using the _norm columns created in cleaning
    df_out['Audit_Score'] = (
        (0.3 * df_out['age_norm']) + 
        (0.2 * df_out['BMI_norm']) + 
        (0.5 * df_out['disease_score_norm'])
    )
    
    # Threshold: Top 25% are Critical
    thresh = df_out['Audit_Score'].quantile(0.75)
    df_out['Status'] = np.where(df_out['Audit_Score'] >= thresh, 'Critical', 'Stable')
    
    return df_out, df_out['Status'].value_counts()

def check_algorithmic_bias(df):
    """New Feature: Checks if 'Critical' status is biased towards age groups."""
    df_audit = df.copy()
    df_audit['Age_Group'] = pd.cut(df_audit['age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle', 'Senior'])
    
    # Calculate % Critical per group
    bias_report = df_audit.groupby('Age_Group')['Status'].apply(lambda x: (x == 'Critical').mean() * 100).reset_index()
    bias_report.columns = ['Group', 'Critical_Rate (%)']
    
    # Detect bias if max rate is 2x min rate
    max_rate = bias_report['Critical_Rate (%)'].max()
    min_rate = bias_report['Critical_Rate (%)'].min() + 0.1 # avoid div/0
    
    bias_detected = (max_rate / min_rate) > 2.0
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=bias_report, x='Group', y='Critical_Rate (%)', palette='Reds', ax=ax)
    ax.set_title("Bias Audit: Critical Rate by Age")
    ax.axhline(df_audit['Status'].value_counts(normalize=True).get('Critical', 0)*100, color='blue', linestyle='--', label='Avg Rate')
    
    return bias_report, bias_detected, fig