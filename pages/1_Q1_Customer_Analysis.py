# 1_Q1_Customer_Analysis.py
import streamlit as st
import pandas as pd
import q1_functions as q1
import time
import streamlit.components.v1 as components

st.set_page_config(page_title="Deep Scan AI", layout="wide")

# --- JAVASCRIPT AUTO-SCROLL ---
def auto_scroll():
    components.html(
        """<script>window.scrollTo({ left: 0, top: document.body.scrollHeight, behavior: "smooth" });</script>""",
        height=0, width=0
    )

st.title(" Deep Scan AI: Advanced Analysis")

# --- SESSION STATE ---
if 'analysis_stage' not in st.session_state:
    st.session_state.analysis_stage = 0
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = None

# --- 1. UPLOADER ---
uploaded_file = st.file_uploader("Upload Analysis File (CSV/Excel)", type=["csv", "xlsx", "xls"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.data_cache = df
        st.success(f"File loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error: {e}")
elif st.button("Load Demo Data"):
    try:
        data = {'Date': pd.date_range(start='1/1/2022', periods=100, freq='W'),
                'Sales': pd.Series(range(100)) + pd.Series([x*0.5 for x in range(100)]),
                'Category': ['A']*50 + ['B']*50}
        st.session_state.data_cache = pd.DataFrame(data)
        st.success("Demo loaded.")
    except:
        st.warning("Demo file missing.")

# --- 2. MAIN PIPELINE ---
if st.session_state.data_cache is not None:
    df = st.session_state.data_cache
    
    # Start Button
    if st.session_state.analysis_stage == 0:
        st.info("System Ready. Initiating AI Grid Processing...")
        if st.button(" Start Deep Scan"):
            st.session_state.analysis_stage = 1
            st.rerun()

    # Progress Tracker
    if st.session_state.analysis_stage > 0:
        steps = ["Sanitation", "EDA", "Statistical Core", "Anomaly Detection", "Predictive Models"]
        cols = st.columns(5)
        for i, step in enumerate(steps):
            if i + 1 < st.session_state.analysis_stage:
                cols[i].success(f" {step}")
            elif i + 1 == st.session_state.analysis_stage:
                cols[i].info(f" {step}...")
            else:
                cols[i].caption(f" {step}")

    # Process Simulator
    def process_batch(stage_num, message):
        if st.session_state.analysis_stage == stage_num:
            prog = st.progress(0)
            stat = st.empty()
            for i in range(2):
                stat.markdown(f"**AI Engine:** {message} ... ({i+1}s)")
                prog.progress((i + 1) * 50)
                time.sleep(1)
            st.session_state.analysis_stage += 1
            st.rerun()

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_container = st.container()

    # --- BATCH 1: CLEANING ---
    if st.session_state.analysis_stage >= 1:
        with row1_col1:
            st.subheader("1. Data Sanitation")
            with st.container(border=True):
                cleaned_df, logs, date_col, stats = q1.clean_data_dynamic(df)
                
                # AI INSIGHT
                prompt = q1.get_data_summary_for_ai("cleaning", cleaned_df, stats)
                ai_insight = q1.query_gpt_api(prompt)
                st.info(f"**AI Audit:**\n\n{ai_insight}")

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.caption("Log Stream")
                    with st.container(height=150):
                        for l in logs: st.text(f"✓ {l}")
                with col_b:
                    st.caption("Cleaned Data Snapshot")
                    st.dataframe(cleaned_df.head(5), height=150, use_container_width=True)
        
        if st.session_state.analysis_stage == 1:
            process_batch(1, "Imputing missing values & normalizing types")

    # --- BATCH 2: EDA ---
    if st.session_state.analysis_stage >= 2:
        with row1_col2:
            st.subheader("2. Exploratory Data Analysis")
            with st.container(border=True):
                # AI INSIGHT
                prompt = q1.get_data_summary_for_ai("eda", cleaned_df)
                ai_insight = q1.query_gpt_api(prompt)
                st.info(f"**AI Distribution Analysis:**\n\n{ai_insight}")

                eda_plots = q1.get_dynamic_eda_plots(cleaned_df)
                if eda_plots:
                    c_p1, c_p2 = st.columns(2)
                    keys = list(eda_plots.keys())
                    for i, k in enumerate(keys):
                        if i < 2:
                            with c_p1 if i == 0 else c_p2:
                                st.pyplot(eda_plots[k])
        
        if st.session_state.analysis_stage == 2:
            auto_scroll()
            process_batch(2, "Generating distribution curves")

    # --- BATCH 3: STATISTICS ---
    if st.session_state.analysis_stage >= 3:
        with row2_col1:
            st.subheader("3. Statistical Core")
            with st.container(border=True):
                # AI INSIGHT
                prompt = q1.get_data_summary_for_ai("stats", cleaned_df)
                ai_insight = q1.query_gpt_api(prompt)
                st.info(f"**AI Statistical Review:**\n\n{ai_insight}")

                stats_df = q1.get_summary_stats(cleaned_df)
                st.dataframe(stats_df, height=250, use_container_width=True)
        
        if st.session_state.analysis_stage == 3:
            auto_scroll()
            process_batch(3, "Calculating variance & deviations")

    # --- BATCH 4: OUTLIERS ---
    if st.session_state.analysis_stage >= 4:
        with row2_col2:
            st.subheader("4. Anomaly Detection")
            with st.container(border=True):
                # AI INSIGHT
                prompt = q1.get_data_summary_for_ai("outliers", cleaned_df)
                ai_insight = q1.query_gpt_api(prompt)
                st.info(f"**AI Risk Assessment:**\n\n{ai_insight}")

                out_plots = q1.get_outlier_plots(cleaned_df)
                if out_plots:
                    keys = list(out_plots.keys())
                    if len(keys) > 0: st.pyplot(out_plots[keys[0]])
        
        if st.session_state.analysis_stage == 4:
            auto_scroll()
            process_batch(4, "Detecting IQR deviations")

    # --- BATCH 5: FORECASTING ---
    if st.session_state.analysis_stage >= 5:
        with row3_container:
            st.markdown("---")
            st.subheader("5. Advanced Predictive Modeling")
            
            if date_col:
                ts_fig, monthly_data = q1.get_dynamic_time_series(cleaned_df, date_col)
                
                col_f1, col_f2 = st.columns([3, 1])
                with col_f2:
                    st.markdown("####  Model Config")
                    # ADDED NEW MODELS HERE
                    models = st.multiselect(
                        "Ensemble Selection",
                        ['Linear Trend', 'Polynomial (Curve)', 'Random Forest', 'ARIMA', 'Exponential Smoothing'],
                        default=['Linear Trend', 'Random Forest']
                    )
                    steps = st.slider("Forecast Horizon (Months)", 3, 24, 12)
                
                with col_f1:
                    if monthly_data is not None and models:
                        f_fig, slope = q1.run_forecasting(monthly_data, steps, models)
                        st.pyplot(f_fig)
                        
                        # AI INSIGHT
                        trend_desc = "Positive Growth" if slope > 0 else "Negative Decline"
                        prompt = q1.get_data_summary_for_ai("forecast", cleaned_df, {'trend': trend_desc})
                        ai_insight = q1.query_gpt_api(prompt)
                        st.success(f"**AI Strategic Outlook:**\n\n{ai_insight}")
            else:
                st.warning("Time-Series Analysis bypassed: No datetime index detected.")

        if st.session_state.analysis_stage == 5:
            auto_scroll()
            process_batch(5, "Finalizing ensemble predictions")

        if st.session_state.analysis_stage >= 6:
            auto_scroll()
            st.success(" Deep Scan Analysis Complete.")
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data Report", csv, "deep_scan_data.csv", "text/csv")

    if st.session_state.analysis_stage > 0:
        st.markdown("---")
        if st.button(" Reset System"):
            st.session_state.analysis_stage = 0
            st.rerun()