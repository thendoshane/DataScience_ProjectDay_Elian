import streamlit as st
import pandas as pd
import q3_functions as q3
import time
import streamlit.components.v1 as components

st.set_page_config(page_title="Q3: Clinical Risk", layout="wide")

def auto_scroll():
    components.html(
        """<script>window.scrollTo({ left: 0, top: document.body.scrollHeight, behavior: "smooth" });</script>""",
        height=0, width=0
    )

st.title("Q3: Clinical Risk Stratification")

# --- STATE ---
if 'risk_stage' not in st.session_state:
    st.session_state.risk_stage = 0
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None

# --- 1. UPLOADER ---
uploaded_file = st.file_uploader("Upload Vitals Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.risk_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.risk_data = pd.read_excel(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error: {e}")
elif st.button("Load Demo Patient Data"):
    # Generate robust demo data
    data = {
        'Patient_ID': range(1001, 1101),
        'BMI_Value': [18 + (x % 30) + (x/10) for x in range(100)],
        'Disease_Severity_Index': [x for x in range(100)],
        'Systolic_BP': [110 + (x % 50) for x in range(100)]
    }
    st.session_state.risk_data = pd.DataFrame(data)
    st.success("Generated Demo Vitals.")

# --- 2. PIPELINE ---
if st.session_state.risk_data is not None:
    df = st.session_state.risk_data
    
    if st.session_state.risk_stage == 0:
        st.info("System Ready. Initiating 5-Stage Clinical Protocol.")
        if st.button(" Start Clinical Scan"):
            st.session_state.risk_stage = 1
            st.rerun()

    # Progress
    if st.session_state.risk_stage > 0:
        steps = ["Vitals Integrity", "Rule-Based Risk", "ML Cohort Analysis", "HTN Feature Eng", "Clinical Summary"]
        cols = st.columns(5)
        for i, step in enumerate(steps):
            if i + 1 < st.session_state.risk_stage:
                cols[i].success(f" {step}")
            elif i + 1 == st.session_state.risk_stage:
                cols[i].info(f" {step}...")
            else:
                cols[i].caption(f" {step}")

    def process_step(stage_num, msg):
        if st.session_state.risk_stage == stage_num:
            prog = st.progress(0)
            stat = st.empty()
            for i in range(2):
                stat.markdown(f"**Medical Engine:** {msg}... ({i+1}s)")
                prog.progress((i+1)*50)
                time.sleep(1)
            st.session_state.risk_stage += 1
            st.rerun()

    # Validate Columns Immediately
    df_clean = q3.validate_columns(df)

    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    row3_container = st.container()

    # ==================================================
    # STAGE 1: VITALS CHECK
    # ==================================================
    if st.session_state.risk_stage >= 1:
        with row1_c1:
            st.subheader("1. Vitals Integrity & Correlations")
            with st.container(border=True):
                # AI Call
                prompt = q3.get_q3_summary_for_ai("vitals", df_clean)
                ai_msg = q3.query_gpt_api(prompt)
                st.info(f"**AI Vitals Assessment:**\n\n{ai_msg}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean BMI", f"{df_clean['BMI'].mean():.1f}")
                c2.metric("Mean BP", f"{df_clean['blood_pressure'].mean():.0f}")
                c3.metric("Records", len(df_clean))

        with row1_c2:
            st.subheader("Correlation Map")
            with st.container(border=True):
                fig_corr = q3.get_correlation_heatmap(df_clean)
                st.pyplot(fig_corr)

        if st.session_state.risk_stage == 1:
            process_step(1, "Mapping columns and scanning correlations")

    # ==================================================
    # STAGE 2: RULE-BASED RISK
    # ==================================================
    if st.session_state.risk_stage >= 2:
        df_risk, counts = q3.classify_risk_logic(df_clean)
        
        with row2_c1:
            st.subheader("2. Rule-Based Triage")
            with st.container(border=True):
                # AI Call
                prompt = q3.get_q3_summary_for_ai("risk_logic", df_risk, counts)
                ai_msg = q3.query_gpt_api(prompt)
                st.warning(f"**AI Triage Protocols:**\n\n{ai_msg}")
                
                st.bar_chart(counts, color=["#FF4B4B"])
                
        if st.session_state.risk_stage == 2:
            auto_scroll()
            process_step(2, "Applying clinical logic ($BMI > 30$)")

    # ==================================================
    # STAGE 3: ML CLUSTERING (New Feature)
    # ==================================================
    if st.session_state.risk_stage >= 3:
        df_clustered = q3.perform_clustering(df_risk)
        
        with row2_c2:
            st.subheader("3. Unsupervised Cohort Discovery")
            


            with st.container(border=True):
                # AI Call
                prompt = q3.get_q3_summary_for_ai("clustering", df_clustered)
                ai_msg = q3.query_gpt_api(prompt)
                st.info(f"**AI Cohort Analysis:**\n\n{ai_msg}")
                
                fig_cluster = q3.get_cluster_plot(df_clustered)
                st.pyplot(fig_cluster)

        if st.session_state.risk_stage == 3:
            auto_scroll()
            process_step(3, "Running K-Means Clustering on Vitals")

    # ==================================================
    # STAGE 4 & 5: FEATURES & SUMMARY
    # ==================================================
    if st.session_state.risk_stage >= 4:
        with row3_container:
            st.markdown("---")
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                st.subheader("4. Hypertension Feature Eng.")
                

                df_final = q3.categorize_bp(df_clustered)
                
                prompt = q3.get_q3_summary_for_ai("hypertension", df_final)
                ai_msg = q3.query_gpt_api(prompt)
                st.success(f"**AI Clinical Impact:**\n\n{ai_msg}")
                
            with c_right:
                st.subheader("5. Consolidated Patient Registry")
                st.dataframe(df_final[['patient_id', 'Risk_Level', 'Cohort_Name', 'BP_Category', 'disease_score']].head(10), use_container_width=True)
                
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("Download Clinical Report", csv, "clinical_risk_analysis.csv", "text/csv")

        if st.session_state.risk_stage == 4:
            auto_scroll()
            #st.balloons()
            st.success(" Protocol Complete.")

    if st.session_state.risk_stage > 0:
        st.markdown("---")
        if st.button(" Reset Protocol"):
            st.session_state.risk_stage = 0
            st.rerun()