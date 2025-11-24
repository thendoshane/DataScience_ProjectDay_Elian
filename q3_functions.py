import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import AzureOpenAI

# --- HARDCODED AZURE CREDENTIALS ---
AZURE_API_KEY = "AgS096Z1phNiJK1KpW0qyQF6vbVqF3dfFwkNUsAJpStzZaswlXEOJQQJ99BKACHYHv6XJ3w3AAAAACOGuFbG"
AZURE_ENDPOINT = "https://thendos-6441-resource.services.ai.azure.com/"
DEPLOYMENT_NAME = "datascience"

# --- ROBUST DATA MAPPING ---
def validate_columns(df):
    """
    Reconstructs dataframe to ensure unique, valid medical columns.
    Prevents 'Duplicate Column' errors.
    """
    # 1. Deduplicate source columns immediately
    df = df.loc[:, ~df.columns.duplicated()]
    source_cols = list(df.columns)
    source_cols_lower = [str(c).lower().strip() for c in source_cols]
    
    # 2. Schema Definition
    schema = {
        'patient_id': ['id', 'patient', 'mrn', 'ref'],
        'BMI': ['bmi', 'body mass', 'obesity'],
        'blood_pressure': ['bp', 'blood pressure', 'systolic', 'pressure'],
        'disease_score': ['score', 'disease', 'severity', 'risk_score', 'index']
    }

    # 3. Construction Strategy
    new_data = {}
    used_indices = set()

    for target, keywords in schema.items():
        found = False
        # Exact match attempt
        for i, col in enumerate(source_cols_lower):
            if col == target.lower() and i not in used_indices:
                new_data[target] = df.iloc[:, i]
                used_indices.add(i)
                found = True
                break
        
        # Keyword match attempt
        if not found:
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
            if target == 'patient_id': new_data[target] = range(1000, 1000+len(df))
            elif target == 'BMI': new_data[target] = 22.0 # Healthy default
            elif target == 'blood_pressure': new_data[target] = 120.0
            elif target == 'disease_score': new_data[target] = 0.0

    df_clean = pd.DataFrame(new_data)
    
    # Force numerics
    for c in ['BMI', 'blood_pressure', 'disease_score']:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce').fillna(0)
        
    return df_clean

# --- AI INTEGRATION ---
def get_q3_summary_for_ai(stage, df, extra_info=None):
    try:
        if stage == "vitals":
            corr = df[['BMI', 'disease_score']].corr().iloc[0,1]
            return (
                f"DATA: Vitals Check. $N={len(df)}$ patients. "
                f"Correlation(BMI, Score) $r = {corr:.2f}$. "
                f"Avg BP $\mu = {df['blood_pressure'].mean():.1f}$. "
                f"Assess the reliability of these vital signs."
            )
        elif stage == "clustering":
            n_clusters = extra_info.get('n_clusters', 3)
            return (
                f"DATA: K-Means Clustering. We identified $k={n_clusters}$ distinct patient cohorts based on BMI, BP, and Score. "
                f"Explain why unsupervised clustering helps identify 'hidden' risk groups."
            )
        elif stage == "risk_logic":
            counts = extra_info
            return (
                f"DATA: Rule-Based Triage. Results: "
                f"High Risk: {counts.get('High Risk', 0)}, Medium: {counts.get('Medium Risk', 0)}. "
                f"Define clinical protocols for the High Risk group."
            )
        elif stage == "hypertension":
            counts = df['BP_Category'].value_counts().to_dict()
            return (
                f"DATA: Hypertension Analysis. Breakdown: {counts}. "
                f"Explain the long-term impact of 'Hypertension S2' on disease progression."
            )
        return "Analyze medical data."
    except:
        return "Analyze medical data."

def query_gpt_api(context_text):
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview", 
            azure_endpoint=AZURE_ENDPOINT
        )
        system_instruction = (
            "You are a Chief Medical Officer. "
            "FORMAT RULES: \n"
            "1. Use ONLY Numbered Lists for protocols.\n"
            "2. Use LaTeX for vitals (e.g., $BP > 140$, $BMI \geq 30$).\n"
            "3. Be extremely concise and clinical."
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

# --- ANALYTICS MODULES ---

def get_correlation_heatmap(df):
    """Stage 1: Heatmap"""
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ['BMI', 'blood_pressure', 'disease_score']
    sns.heatmap(df[cols].corr(), annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title("Vitals Correlation Matrix")
    return fig

def perform_clustering(df):
    """Stage 3: K-Means"""
    df_ml = df[['BMI', 'blood_pressure', 'disease_score']].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_ml)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Map clusters to readable names based on severity
    cluster_means = df.groupby('Cluster')['disease_score'].mean().sort_values()
    mapping = {
        cluster_means.index[0]: 'Cohort A (Low Severity)',
        cluster_means.index[1]: 'Cohort B (Moderate)',
        cluster_means.index[2]: 'Cohort C (Critical)'
    }
    df['Cohort_Name'] = df['Cluster'].map(mapping)
    return df

def get_cluster_plot(df):
    """Visualizes Clusters"""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=df, x='BMI', y='disease_score', 
        hue='Cohort_Name', style='Cohort_Name', 
        palette='viridis', s=100, ax=ax
    )
    ax.set_title("Unsupervised Patient Cohorts (K-Means)")
    ax.set_xlabel("BMI Index")
    ax.set_ylabel("Disease Score")
    return fig

def classify_risk_logic(df):
    """Stage 2: Rule Based"""
    df_out = df.copy()
    levels = []
    for _, row in df_out.iterrows():
        s, b, p = row['disease_score'], row['BMI'], row['blood_pressure']
        if s > 75 and b > 30: levels.append("High Risk")
        elif (s > 50 and b > 25) or p > 140: levels.append("Medium Risk")
        else: levels.append("Low Risk")
    df_out['Risk_Level'] = levels
    return df_out, df_out['Risk_Level'].value_counts()

def categorize_bp(df):
    """Stage 4: Feature Eng"""
    df_out = df.copy()
    bins = [0, 120, 130, 140, 180, 999]
    labels = ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN', 'Crisis']
    df_out['BP_Category'] = pd.cut(df_out['blood_pressure'], bins=bins, labels=labels)
    return df_out