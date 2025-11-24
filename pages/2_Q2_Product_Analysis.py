import streamlit as st
import pandas as pd
import q2_functions as q2
import time
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(page_title="Q2: Product Strategy", layout="wide")

def auto_scroll():
    components.html(
        """<script>window.scrollTo({ left: 0, top: document.body.scrollHeight, behavior: "smooth" });</script>""",
        height=0, width=0
    )

st.title(" Q2: Product Portfolio & Revenue Strategy")

# --- STATE ---
if 'prod_stage' not in st.session_state:
    st.session_state.prod_stage = 0
if 'prod_data' not in st.session_state:
    st.session_state.prod_data = None

# --- 1. UPLOADER ---
uploaded_file = st.file_uploader("Upload Product Inventory (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.prod_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.prod_data = pd.read_excel(uploaded_file)
        st.success(f"Inventory Loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error: {e}")
elif st.button("Load Sample Inventory"):
    try:
        # Generate robust demo data
        np.random.seed(42)
        rows = 100
        data = {
            'Product_Name': [f"SKU_{i}" for i in range(1, rows+1)],
            'Category_Group': np.random.choice(['Electronics', 'Fashion', 'Home', 'Beauty'], rows),
            'Unit_Cost': np.random.uniform(5, 500, rows),
            'Qty_Sold': np.random.randint(1, 200, rows)
        }
        # Introduce missing revenue to test calculation logic
        df_demo = pd.DataFrame(data)
        df_demo.loc[0:10, 'Unit_Cost'] = np.nan # Test missing price
        st.session_state.prod_data = df_demo
        st.success("Sample inventory generated.")
    except Exception as e:
        st.error(f"Failed to load sample: {e}")

# --- 2. PIPELINE ---
if st.session_state.prod_data is not None:
    df = st.session_state.prod_data
    
    if st.session_state.prod_stage == 0:
        st.info("System Ready. Click to run Product Intelligence Scan.")
        if st.button(" Start Product Audit"):
            st.session_state.prod_stage = 1
            st.rerun()

    # Progress
    if st.session_state.prod_stage > 0:
        steps = ["Data Standardization", "Pareto & Elasticity", "Performance Audits", "Code Handover"]
        cols = st.columns(4)
        for i, step in enumerate(steps):
            if i + 1 < st.session_state.prod_stage:
                cols[i].success(f" {step}")
            elif i + 1 == st.session_state.prod_stage:
                cols[i].info(f" {step}...")
            else:
                cols[i].caption(f" {step}")

    def process_step(stage_num, msg):
        if st.session_state.prod_stage == stage_num:
            prog = st.progress(0)
            stat = st.empty()
            for i in range(2):
                stat.markdown(f"**Strategy Engine:** {msg}... ({i+1}s)")
                prog.progress((i+1)*50)
                time.sleep(1)
            st.session_state.prod_stage += 1
            st.rerun()

    # Validate Columns (Silent background process)
    df_mapped = q2.validate_columns(df)

    # --- GRID LAYOUT ---
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    row3_container = st.container()

    # ==================================================
    # STAGE 1: REVENUE & SANITATION
    # ==================================================
    if st.session_state.prod_stage >= 1:
        # Calculate once
        df_calc, missing_count = q2.calculate_revenue(df_mapped)
        
        with row1_c1:
            st.subheader("1. Financial Standardization")
            with st.container(border=True):
                # AI Call
                prompt = q2.get_q2_summary_for_ai("revenue", df_calc, {'missing_revenue': missing_count})
                ai_msg = q2.query_gpt_api(prompt)
                st.info(f"**AI Audit:**\n\n{ai_msg}")
                
                c1, c2 = st.columns(2)
                c1.metric("Total Revenue", f"${df_calc['revenue'].sum():,.0f}")
                c2.metric("Data Repaired", f"{missing_count} rows")
                
                st.caption("Standardized Data Snapshot")
                st.dataframe(df_calc[['product_name', 'price', 'units_sold', 'revenue']].head(), use_container_width=True)

        if st.session_state.prod_stage == 1:
            process_step(1, "Mapping schema and imputing financials")

    # ==================================================
    # STAGE 2: PARETO & ELASTICITY (New Feature)
    # ==================================================
    if st.session_state.prod_stage >= 2:
        with row1_c2:
            st.subheader("2. Market Dynamics (Pareto)")
            with st.container(border=True):
                t20, tot, df_sorted = q2.perform_pareto_analysis(df_calc)
                
                # AI Call
                prompt = q2.get_q2_summary_for_ai("pareto", df_calc, {'top_20_revenue': t20, 'total_revenue': tot})
                ai_msg = q2.query_gpt_api(prompt)
                st.info(f"**AI Concentration Analysis:**\n\n{ai_msg}")

                pareto_fig = q2.get_pareto_plot(df_calc)
                st.pyplot(pareto_fig)

        if st.session_state.prod_stage == 2:
            auto_scroll()
            process_step(2, "Calculating 80/20 concentration & pricing sensitivity")

    # ==================================================
    # STAGE 3: PERFORMANCE AUDIT (Split View)
    # ==================================================
    if st.session_state.prod_stage >= 3:
        # Top Performers vs Elasticity
        with row2_c1:
            st.subheader("3. Pricing Elasticity")
            with st.container(border=True):
                # AI Call
                prompt = q2.get_q2_summary_for_ai("elasticity", df_calc)
                ai_msg = q2.query_gpt_api(prompt)
                st.info(f"**AI Pricing Strategy:**\n\n{ai_msg}")
                
                elast_fig = q2.get_elasticity_plot(df_calc)
                st.pyplot(elast_fig)

        with row2_c2:
            st.subheader("Inventory Health")
            with st.container(border=True):
                underperformers = q2.flag_underperformers(df_calc)
                
                # AI Call
                prompt = q2.get_q2_summary_for_ai("underperformers", underperformers)
                ai_msg = q2.query_gpt_api(prompt)
                st.warning(f"**AI Liquidation Advice:**\n\n{ai_msg}")

                if not underperformers.empty:
                    st.dataframe(underperformers[['product_name', 'revenue', 'units_sold']].head(), use_container_width=True)
                else:
                    st.success("Clean Inventory! No dead stock detected.")

        if st.session_state.prod_stage == 3:
            auto_scroll()
            process_step(3, "Isolating dead stock & high performers")

    # ==================================================
    # STAGE 4: EXPORT & CODE
    # ==================================================
    if st.session_state.prod_stage >= 4:
        with row3_container:
            st.markdown("---")
            st.subheader("4. Deployment Handoff")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Automated Pipeline Code:**")
                st.code(q2.get_reusable_function_code(), language='python')
            with c2:
                st.success("Optimization Complete.")
                st.info("The code on the left is production-ready for your weekly automated jobs.")
                csv = df_calc.to_csv(index=False).encode('utf-8')
                st.download_button("Download Optimized Data", csv, "q2_product_strategy.csv", "text/csv")

        if st.session_state.prod_stage == 4:
            auto_scroll()
            st.success(" Product Strategy Scan Finished.")

    if st.session_state.prod_stage > 0:
        st.markdown("---")
        if st.button(" Reset System"):
            st.session_state.prod_stage = 0
            st.rerun()