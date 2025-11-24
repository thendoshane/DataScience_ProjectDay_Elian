import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from openai import AzureOpenAI

# --- HARDCODED AZURE CREDENTIALS ---
AZURE_API_KEY = "AgS096Z1phNiJK1KpW0qyQF6vbVqF3dfFwkNUsAJpStzZaswlXEOJQQJ99BKACHYHv6XJ3w3AAAAACOGuFbG"
AZURE_ENDPOINT = "https://thendos-6441-resource.services.ai.azure.com/"
DEPLOYMENT_NAME = "datascience"

# --- ROBUST DATA MAPPING (THE FIX) ---
def validate_columns(df):
    """
    Reconstructs the dataframe to ensure strictly unique columns.
    Drops duplicates and handles fuzzy matching safely.
    """
    # 1. First, strictly remove duplicate columns from the SOURCE file
    # (e.g., if user uploaded a file with two 'Price' columns)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 2. Normalize source columns for matching
    source_cols = list(df.columns)
    source_cols_lower = [str(c).lower().strip() for c in source_cols]
    
    # 3. Define target schema with strict keyword priority
    #    The order of keywords matters! Specific terms first.
    schema = {
        'revenue': ['revenue', 'total sales', 'total_sales', 'turnover', 'sales amount', 'amount', 'sales'],
        'units_sold': ['units_sold', 'units', 'quantity', 'qty', 'count', 'volume', 'sold'],
        'price': ['price', 'unit_price', 'unit_cost', 'cost', 'rate', 'msrp'],
        'category': ['category', 'department', 'dept', 'group', 'segment', 'class'],
        'product_name': ['product_name', 'product', 'item_name', 'item', 'description', 'sku', 'desc', 'name']
    }

    # 4. "Select and Construct" Strategy
    #    We build a new DF column by column, marking source cols as 'used' to prevent double-dipping.
    new_data = {}
    used_source_indices = set()

    for target, keywords in schema.items():
        found = False
        
        # Priority A: Check for Exact Match in Source
        for i, col_name in enumerate(source_cols_lower):
            if col_name == target and i not in used_source_indices:
                new_data[target] = df.iloc[:, i]
                used_source_indices.add(i)
                found = True
                break
        
        # Priority B: Check for Keyword Containment
        if not found:
            for kw in keywords:
                for i, col_name in enumerate(source_cols_lower):
                    # Check if keyword is in column name AND column hasn't been used
                    if kw in col_name and i not in used_source_indices:
                        new_data[target] = df.iloc[:, i]
                        used_source_indices.add(i)
                        found = True
                        break
                if found: break
        
        # Priority C: Fill with Default/NaN if not found
        if not found:
            if target in ['revenue', 'price']:
                new_data[target] = np.nan
            elif target == 'units_sold':
                new_data[target] = 0
            elif target == 'category':
                new_data[target] = "General"
            elif target == 'product_name':
                new_data[target] = [f"Item_{x}" for x in range(len(df))]

    # 5. Create final clean DataFrame (Guaranteed Unique Columns)
    df_clean = pd.DataFrame(new_data)
    
    # 6. Force Numeric Types safely
    for c in ['price', 'units_sold', 'revenue']:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    return df_clean

# --- AI INTEGRATION ---
def get_q2_summary_for_ai(stage, df, extra_info=None):
    try:
        if stage == "revenue":
            missing = extra_info.get('missing_revenue', 0)
            total_rev = df['revenue'].sum()
            avg_rev = df['revenue'].mean()
            return (
                f"DATA: Revenue Calc. Total Revenue $R_{{tot}} = {total_rev:,.2f}$. "
                f"Avg Revenue per SKU $\mu = {avg_rev:,.2f}$. "
                f"Imputed {missing} missing values using $P \\times Q$. "
                f"Analyze the stability of this revenue base."
            )
        elif stage == "pareto":
            top_20_pct_rev = extra_info.get('top_20_revenue', 0)
            total_rev = extra_info.get('total_revenue', 1)
            concentration = (top_20_pct_rev / total_rev) * 100
            return (
                f"DATA: Pareto Analysis. The top 20% of products drive ${concentration:.1f}\\%$ of total revenue. "
                f"Standard Pareto is 80/20. "
                f"Evaluate the 'Concentration Risk' based on this percentage."
            )
        elif stage == "elasticity":
            # Safety check for correlation
            if len(df) > 1 and df['price'].std() > 0 and df['units_sold'].std() > 0:
                corr = df['price'].corr(df['units_sold'])
            else:
                corr = 0
            return (
                f"DATA: Price Elasticity. Correlation between Price ($P$) and Volume ($Q$) is $r = {corr:.2f}$. "
                f"If $r < 0$, demand is elastic (normal). If $r > 0$, it is a Giffen/Veblen good anomaly. "
                f"Interpret this pricing power."
            )
        elif stage == "underperformers":
            count = len(df)
            return (
                f"DATA: Dead Stock. Found $N={count}$ SKUs with Sales $< 50$ AND Revenue $< $1000. "
                f"Suggest liquidation strategies (e.g., Bundling, Clearance) to free up working capital."
            )
        return "Analyze product data."
    except Exception as e:
        return f"Error building prompt: {e}"

def query_gpt_api(context_text):
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview", 
            azure_endpoint=AZURE_ENDPOINT
        )
        system_instruction = (
            "You are a Retail Strategy Expert. "
            "FORMAT RULES: \n"
            "1. Use ONLY Numbered Lists.\n"
            "2. Use LaTeX math symbols where possible (e.g., $R_{tot}$, $\mu$, $\\Delta$, %). \n"
            "3. Be concise and business-focused (Max 100 words)."
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

# --- CALCULATION LOGIC ---
def calculate_revenue(df):
    try:
        df_calc = df.copy()
        
        # Fill missing prices with Category Median
        if 'category' in df_calc.columns:
            df_calc['price'] = df_calc['price'].fillna(df_calc.groupby('category')['price'].transform('median'))
        
        df_calc['price'] = df_calc['price'].fillna(df_calc['price'].median()) # Global fallback
        if pd.isna(df_calc['price'].median()): df_calc['price'] = 0 # Ultimate fallback

        missing_count = df_calc['revenue'].isnull().sum()
        mask = df_calc['revenue'].isnull()
        df_calc.loc[mask, 'revenue'] = df_calc['units_sold'].fillna(0) * df_calc['price'].fillna(0)
        
        return df_calc, missing_count
    except Exception as e:
        # Return original with 0s to prevent app crash
        df['revenue'] = 0
        return df, 0

def perform_pareto_analysis(df):
    try:
        df_sorted = df.sort_values(by='revenue', ascending=False)
        total_revenue = df_sorted['revenue'].sum()
        top_20_count = int(max(1, len(df) * 0.2)) # Ensure at least 1 item
        top_20_revenue = df_sorted['revenue'].head(top_20_count).sum()
        return top_20_revenue, total_revenue, df_sorted
    except:
        return 0, 1, df

def flag_underperformers(df):
    try:
        df_flag = df.copy()
        units_thresh = df_flag['units_sold'].quantile(0.25)
        rev_thresh = df_flag['revenue'].quantile(0.25)
        
        # Safety defaults if data is uniform/zero
        if np.isnan(units_thresh) or units_thresh == 0: units_thresh = 10
        if np.isnan(rev_thresh) or rev_thresh == 0: rev_thresh = 100
        
        df_flag['is_underperforming'] = (df_flag['units_sold'] <= units_thresh) & (df_flag['revenue'] <= rev_thresh)
        underperformers = df_flag[df_flag['is_underperforming'] == True]
        return underperformers
    except:
        return pd.DataFrame() # Empty result on error

def get_top_performers(df, n=5):
    try:
        return df.nlargest(n, 'revenue')
    except:
        return df.head(n)

# --- PLOTTING ---
def get_pareto_plot(df):
    try:
        df_sorted = df.sort_values(by='revenue', ascending=False).reset_index(drop=True)
        # Avoid division by zero
        total_rev = df_sorted['revenue'].sum()
        if total_rev == 0: total_rev = 1 
            
        df_sorted['cumulative_revenue'] = df_sorted['revenue'].cumsum()
        df_sorted['cumulative_percentage'] = 100 * df_sorted['cumulative_revenue'] / total_rev
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_sorted.index, df_sorted['cumulative_percentage'], color='#d62728', linewidth=2)
        ax.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80% Revenue')
        ax.set_title("Pareto Curve (Concentration Risk)")
        ax.set_ylabel("Cumulative Revenue (%)")
        ax.set_xlabel("Products Rank")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center')
        return fig

def get_elasticity_plot(df):
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Handle cases with 0 or negative values for log scale
        plot_df = df[df['price'] > 0].copy()
        plot_df = plot_df[plot_df['units_sold'] > 0]
        
        if not plot_df.empty:
            sns.scatterplot(data=plot_df, x='price', y='units_sold', hue='category', alpha=0.7, ax=ax)
            ax.set_title("Price Elasticity (Price vs Volume)")
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
             ax.text(0.5, 0.5, "Insufficient positive data for log plot", ha='center')
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Plot Error", ha='center')
        return fig

def get_reusable_function_code():
    code_text = """
def optimize_inventory(df):
    '''
    Production-grade inventory optimizer.
    1. Imputes missing revenue using category medians.
    2. Flags dead stock (bottom 25% quantile).
    3. Calculates 80/20 Pareto concentration.
    '''
    # Revenue Logic
    df['price'] = df['price'].fillna(df.groupby('category')['price'].transform('median'))
    df['revenue'] = df['revenue'].fillna(df['price'] * df['units_sold'])
    
    # Pareto Logic
    df = df.sort_values('revenue', ascending=False)
    top_20_pct = df.head(int(len(df)*0.2))['revenue'].sum() / df['revenue'].sum()
    
    # Dead Stock Logic
    low_sales = df['units_sold'] < df['units_sold'].quantile(0.25)
    low_rev = df['revenue'] < df['revenue'].quantile(0.25)
    dead_stock = df[low_sales & low_rev]
    
    return dead_stock, top_20_pct
    """
    return code_text