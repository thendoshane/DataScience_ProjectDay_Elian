import streamlit as st
import pandas as pd
import q4_functions as q4
import time
import streamlit.components.v1 as components

st.set_page_config(page_title="Q4: Health Audit", layout="wide")

def auto_scroll():
    components.html(
        """<script>window.scrollTo({ left: 0, top: document.body.scrollHeight, behavior: "smooth" });</script>""",
        height=0, width=0
    )

st.title(" Q4: Eastern Cape Health Audit System")

# --- STATE ---
if 'audit_stage' not in st.session_state:
    st.session_state.audit_stage = 0
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None

# --- 1. UPLOADER ---
uploaded_file = st.file_uploader("Upload Patient Records (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.audit_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.audit_data = pd.read_excel(uploaded_file)
        st.success(f"File Access: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error: {e}")
elif st.button("Generate Audit Sample"):
    # Robust Dummy Data
    data = {
        'Ref_ID': range(1001, 1101),
        'Patient_Age': [x for x in range(20, 80)] + [x for x in range(30, 70)],
        'Severity_Index': [x * 0.9 for x in range(100)],
        'Body_Mass_Index': [20 + (x%15) for x in range(100)]
    }
    st.session_state.audit_data = pd.DataFrame(data)
    st.success("Generated random clinic sample.")

# --- 2. PIPELINE ---
if st.session_state.audit_data is not None:
    df = st.session_state.audit_data
    
    if st.session_state.audit_stage == 0:
        st.info("System Ready. Initializing 5-Point Audit Protocol.")
        if st.button(" Start Audit Cycle"):
            st.session_state.audit_stage = 1
            st.rerun()

    # Progress
    if st.session_state.audit_stage > 0:
        steps = ["Data Hygiene", "Validation Sampling", "Demographics", "Risk Algorithm", "Bias Audit"]
        cols = st.columns(5)
        for i, step in enumerate(steps):
            if i + 1 < st.session_state.audit_stage:
                cols[i].success(f" {step}")
            elif i + 1 == st.session_state.audit_stage:
                cols[i].info(f" {step}...")
            else:
                cols[i].caption(f" {step}")

    def process_step(stage_num, msg):
        if st.session_state.audit_stage == stage_num:
            prog = st.progress(0)
            status = st.empty()
            for i in range(2):
                status.markdown(f"**Auditor Engine:** {msg}... ({i+1}s)")
                prog.progress((i+1)*50)
                time.sleep(1)
            st.session_state.audit_stage += 1
            st.rerun()

    # Clean Data Immediately
    df_clean = q4.validate_columns(df)

    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    row3_container = st.container()

    # ==================================================
    # STAGE 1: HYGIENE
    # ==================================================
    if st.session_state.audit_stage >= 1:
        with row1_c1:
            st.subheader("1. Data Hygiene (Sanitization)")
            with st.container(border=True):
                cleaned_df, dups = q4.clean_audit_data(df_clean)
                
                # AI Insight
                prompt = q4.get_audit_summary_for_ai("cleaning", cleaned_df, {'duplicates': dups})
                ai_msg = q4.query_gpt_api(prompt)
                st.info(f"**AI Auditor:**\n\n{ai_msg}")

                c1, c2 = st.columns(2)
                c1.metric("Valid Records", len(cleaned_df))
                c2.metric("Duplicates Dropped", dups)
                
                st.caption("Normalized Data Stream")
                st.dataframe(cleaned_df[['patient_id', 'age', 'disease_score', 'age_norm']].head(), height=150, use_container_width=True)

        if st.session_state.audit_stage == 1:
            process_step(1, "Normalizing variables and purging duplicates")

    # ==================================================
    # STAGE 2: SAMPLING
    # ==================================================
    if st.session_state.audit_stage >= 2:
        with row1_c2:
            st.subheader("2. Statistical Sampling ($n=20$)")
            with st.container(border=True):
                full_stats, sample_stats = q4.get_random_sample_comparison(cleaned_df)
                
                prompt = q4.get_audit_summary_for_ai("sampling", cleaned_df)
                ai_msg = q4.query_gpt_api(prompt)
                st.info(f"**AI Sampling Review:**\n\n{ai_msg}")

                tab_a, tab_b = st.tabs(["Sample Stats", "Population Stats"])
                with tab_a: st.dataframe(sample_stats, height=150, use_container_width=True)
                with tab_b: st.dataframe(full_stats, height=150, use_container_width=True)

        if st.session_state.audit_stage == 2:
            auto_scroll()
            process_step(2, "Running Monte Carlo validation")

    # ==================================================
    # STAGE 3: DISTRIBUTION
    # ==================================================
    if st.session_state.audit_stage >= 3:
        with row2_c1:
            st.subheader("3. Demographic Distribution")
            with st.container(border=True):
                # AI Insight handled in prompt above or generic
                st.markdown("**Population Age Structure:**")
                dist_fig = q4.get_age_distribution(cleaned_df)
                st.pyplot(dist_fig)

        if st.session_state.audit_stage == 3:
            auto_scroll()
            process_step(3, "Calculating density frequencies")

    # ==================================================
    # STAGE 4: CLASSIFICATION
    # ==================================================
    if st.session_state.audit_stage >= 4:
        class_df, counts = q4.classify_patients_q4(cleaned_df)
        
        with row2_c2:
            st.subheader("4. Risk Triage Algorithm")
            with st.container(border=True):
                prompt = q4.get_audit_summary_for_ai("classification", cleaned_df, counts.to_dict())
                ai_msg = q4.query_gpt_api(prompt)
                st.info(f"**AI Triage Assessment:**\n\n{ai_msg}")

                st.bar_chart(counts, color="#d62728")

        if st.session_state.audit_stage == 4:
            auto_scroll()
            process_step(4, "Computing Weighted Audit Scores")

    # ==================================================
    # STAGE 5: ETHICS & BIAS (New Feature)
    # ==================================================
    if st.session_state.audit_stage >= 5:
        with row3_container:
            st.markdown("---")
            st.subheader("5. Algorithmic Bias Audit (Ethical AI)")
            
            bias_report, is_biased, fig_bias = q4.check_algorithmic_bias(class_df)
            
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                
                if is_biased:
                    st.error(" FAIL: Bias Detected.")
                    st.write("The algorithm assigns 'Critical' status disproportionately to specific age groups.")
                else:
                    st.success(" PASS: No Significant Bias.")
                    st.write("Risk distribution appears equitable across age cohorts.")
                
                # AI Bias Insight
                prompt = q4.get_audit_summary_for_ai("bias", class_df, {'bias_detected': is_biased})
                ai_msg = q4.query_gpt_api(prompt)
                st.markdown(f"**AI Ethical Review:** {ai_msg}")

            with c_right:
                st.pyplot(fig_bias)
                
            st.markdown("---")
            #st.balloons()
            st.success(" Audit Cycle Complete.")
            
            csv = class_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Audited Records", csv, "final_health_audit.csv", "text/csv")
            auto_scroll()

    if st.session_state.audit_stage > 0:
        st.markdown("---")
        if st.button(" Reset Audit"):
            st.session_state.audit_stage = 0
            st.rerun()