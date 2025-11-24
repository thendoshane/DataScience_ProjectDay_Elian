# q1_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from openai import AzureOpenAI

# --- HARDCODED AZURE CREDENTIALS ---
AZURE_API_KEY = "AgS096Z1phNiJK1KpW0qyQF6vbVqF3dfFwkNUsAJpStzZaswlXEOJQQJ99BKACHYHv6XJ3w3AAAAACOGuFbG"
AZURE_ENDPOINT = "https://thendos-6441-resource.services.ai.azure.com/"
DEPLOYMENT_NAME = "datascience"

# --- AI INTEGRATION ---

def get_data_summary_for_ai(stage, df, extra_info=None):
    """
    Creates a text summary.
    """
    summary = ""
    if stage == "cleaning":
        stats = extra_info
        summary = (
            f"DATA: Cleaning Stage. Rows: $N={len(df)}$. "
            f"Duplicates: ${stats.get('duplicates', 0)}$. "
            f"Imputed: ${stats.get('missing_filled', 0)}$. "
            f"Date Col: {stats.get('date_col', 'None')}."
        )
    
    elif stage == "eda":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary = (
            f"DATA: EDA. Numeric: {num_cols[:5]}. "
            f"Analyze distribution shapes (Normal/Skewed). "
            f"Check for multi-modal distributions."
        )

    elif stage == "stats":
        desc = df.describe().to_string()
        summary = (
            f"DATA: Statistics. Summary Matrix:\n{desc}\n"
            f"Analyze $\mu$ (mean) vs median for skewness. "
            f"Identify high variance $\sigma^2$."
        )

    elif stage == "outliers":
        summary = (
            f"DATA: Outliers. "
            f"Evaluate impact of extreme values $> 1.5 * IQR$. "
            f"Recommend removal or capping strategies."
        )
        
    elif stage == "forecast":
        trend = extra_info.get('trend', 'unknown')
        summary = (
            f"DATA: Forecasting. Trend detected: {trend}. "
            f"Models used: Linear ($y=mx+c$), ARIMA, Random Forest. "
            f"Provide strategic growth tactics based on $dy/dx$ trajectory."
        )

    return summary

def query_gpt_api(context_text):
    """
    Sends the context to Azure OpenAI API with strict formatting rules.
    """
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview", 
            azure_endpoint=AZURE_ENDPOINT
        )

        # STRICT SYSTEM PROMPT
        system_instruction = (
            "You are a Lead Data Scientist. "
            "FORMAT RULES: \n"
            "1. Use ONLY Numbered Lists.\n"
            "2. Use LaTeX math symbols where possible (e.g., $\mu$, $\sigma$, $R^2$, $\Delta$).\n"
            "3. Be extremely concise. No introductory fluff.\n"
            "4. Focus on deep analytical alignment and business logic."
        )

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME, 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": context_text}
            ],
            temperature=0.5, # Lower temperature for more structured/analytical output
            max_tokens=400 
        )

        return response.choices[0].message.content

    except Exception as e:
        return f" **Azure API Error:** {str(e)}"

# --- BATCH 1: CLEANING ---
def clean_data_dynamic(df):
    df_clean = df.copy()
    log = []
    duplicates_removed = 0
    missing_filled_count = 0
    
    # 1. Duplicates
    initial = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial - len(df_clean)
    if duplicates_removed > 0:
        log.append(f"Removed {duplicates_removed} duplicate rows.")

    # 2. Detect Date
    date_col_name = None
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                temp_col = pd.to_datetime(df_clean[col], errors='coerce')
                if temp_col.notna().sum() > 0.8 * len(df_clean):
                    df_clean[col] = temp_col
                    date_col_name = col
                    log.append(f"Converted '{col}' to Datetime.")
                    break 
            except:
                pass

    # 3. Fill Missing
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_filled_count += df_clean[col].isnull().sum()
            mean_val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(mean_val)
            log.append(f"Filled numeric '{col}' with $\mu$ ({mean_val:.2f}).")

    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_filled_count += df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna("Unknown")
            log.append(f"Filled text '{col}' with 'Unknown'.")

    stats = {
        'duplicates': duplicates_removed, 
        'missing_filled': missing_filled_count,
        'date_col': date_col_name
    }
    return df_clean, log, date_col_name, stats

# --- BATCH 2: DISTRIBUTIONS ---
def get_dynamic_eda_plots(df):
    plots = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(num_cols[:2]):
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[col], kde=True, ax=ax, color='#2a9d8f')
        ax.set_title(f'Dist: {col}', fontsize=10)
        plots[f'num_{i}'] = fig

    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns 
                if df[c].nunique() < 20 and df[c].nunique() > 1]
    for i, col in enumerate(cat_cols[:2]):
        fig, ax = plt.subplots(figsize=(5, 3))
        top_vals = df[col].value_counts().nlargest(10)
        sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax, palette='mako')
        ax.set_title(f'Counts: {col}', fontsize=10)
        plots[f'cat_{i}'] = fig
    return plots

# --- BATCH 3: STATISTICS ---
def get_summary_stats(df):
    return df.describe(include='all').T

# --- BATCH 4: OUTLIERS ---
def get_outlier_plots(df):
    plots = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(num_cols[:2]):
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.boxplot(x=df[col], ax=ax, color='#e9c46a')
        ax.set_title(f'IQR Outliers: {col}', fontsize=10)
        plots[f'out_{i}'] = fig
    return plots

# --- BATCH 5: ADVANCED FORECASTING ---
def get_dynamic_time_series(df, date_col):
    if date_col:
        monthly_counts = df.set_index(date_col).resample('M').size().rename('Records')
        rolling = monthly_counts.rolling(window=3).mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly_counts.index, monthly_counts, label='Actual $y$', alpha=0.6)
        ax.plot(rolling.index, rolling, label='MA (Window=3)', color='red', linestyle='--')
        ax.set_title(f"Time Series $f(t)$ ({date_col})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, monthly_counts
    return None, None

def run_forecasting(monthly_data, steps, selected_models):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Base Data
    ax.plot(monthly_data.index, monthly_data, label='Historical', color='black', linewidth=2, alpha=0.5)
    last_date = monthly_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='M')
    
    # Pre-calc arrays
    X = np.arange(len(monthly_data)).reshape(-1, 1)
    y = monthly_data.values
    future_X = np.arange(len(monthly_data), len(monthly_data) + steps).reshape(-1, 1)
    
    # 1. Linear
    if 'Linear Trend' in selected_models:
        model = LinearRegression()
        model.fit(X, y)
        forecast = model.predict(future_X)
        forecast = np.maximum(forecast, 0) # No negative values
        ax.plot(future_dates, forecast, label='Linear $y=mx+c$', linestyle='--')

    # 2. Polynomial (Curved)
    if 'Polynomial (Curve)' in selected_models:
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X, y)
        forecast = model.predict(future_X)
        forecast = np.maximum(forecast, 0)
        ax.plot(future_dates, forecast, label='Poly Degree 2', linestyle='--', color='purple')

    # 3. Random Forest (Non-Linear)
    if 'Random Forest' in selected_models:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        forecast = model.predict(future_X)
        ax.plot(future_dates, forecast, label='Random Forest (Ensemble)', linestyle='-.', color='green')
        
    # 4. ARIMA
    if 'ARIMA' in selected_models:
        try:
            model = ARIMA(monthly_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            ax.plot(future_dates, forecast, label='ARIMA $(p,d,q)$', linestyle=':', color='orange')
        except:
            pass
            
    # 5. Exponential Smoothing
    if 'Exponential Smoothing' in selected_models:
        try:
            model = ExponentialSmoothing(monthly_data, trend='add', seasonal=None).fit()
            forecast = model.forecast(steps)
            ax.plot(future_dates, forecast, label='Exp. Smoothing', linestyle=':')
        except:
            pass

    ax.set_title("Multi-Model Predictive Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    slope = 0
    if len(monthly_data) > 1:
        slope = monthly_data.iloc[-1] - monthly_data.iloc[0]
        
    return fig, slope