import streamlit as st
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Eduvos Data Science Toolkit",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; 
        font-weight: 700; 
        color: #003f5c; 
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.5rem; 
        color: #58508d; 
        font-weight: 400;
    }
    .card-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff6361;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .hero-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
col_logo, col_title = st.columns([1, 3])

with col_logo:
    # Placeholder for a logo or relevant icon
    st.markdown("##  **DS-44**")
    st.caption("Module: ITPFA0-44")

with col_title:
    st.markdown('<div class="main-header">Eduvos Data Science Toolkit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Python for Data Science | Comprehensive Assessment Project</div>', unsafe_allow_html=True)

st.markdown("---")

# --- THE "DATA AGNOSTIC" PITCH ---
st.markdown("###  Universal Data Adapter")
st.markdown("""
<div class="hero-text">
1. Project Overview

- Real-World Scenarios: The project consists of four practical case studies where you apply Python for Data Science to solve problems across different industries.

- Data Cleaning & Preparation: Each section requires you to clean messy datasets, fix missing or inconsistent values, extract meaningful features, and prepare data for analysis.

- Data Analysis & Insights: You must explore the data, identify trends or patterns, perform calculations, segment groups, and interpret real-world implications.

- Visualizations: The project requires charts and visuals to support your findings, helping you communicate insights clearly.

- Classification & Modelling: Some sections involve creating your own logic for risk levels or product flags, and even applying Naïve Bayes-style classification.

- Real-World Context: The four datasets span customer analytics, retail product performance, rural healthcare risk assessment, and provincial health audits.

- Ethical Reflection: You must consider fairness, privacy, and responsible use of data, especially in health-related tasks.

- Deliverables: For each scenario, you produce cleaned datasets, visualizations, classification outputs, and Python scripts demonstrating automation and analysis.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- NAVIGATION HUB (THE 4 QUESTIONS) ---
st.subheader(" Operational Modules")
st.caption("Select a module below to launch the specific analytical engine.")

# ROW 1: Q1 & Q2
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("### 1️ Customer Intelligence (GlobeStream)")
        st.markdown("**Focus:** Time-Series, Data Cleaning, Segmentation")
        st.info("Analyze customer growth, clean messy date formats, and forecast future sign-ups.")
        
        st.markdown("**Use this if you want to:**")
        st.markdown("- Clean messy `subscription_date` columns.")
        st.markdown("- Forecast customer acquisition trends.")
        st.markdown("- Segment users by Age/Region.")
        
        if st.button("Launch Q1: Customer Scan", use_container_width=True):
            st.switch_page("pages/1_Q1_Customer_Analysis.py")

with c2:
    with st.container(border=True):
        st.markdown("### 2️ Product Strategy (NovaMart)")
        st.markdown("**Focus:** Revenue Calc, Pareto (80/20), Elasticity")
        st.success("Identify 'Dead Stock', calculate true revenue, and find your top 20% products.")
        
        st.markdown("**Use this if you want to:**")
        st.markdown("- Fix missing `Price` or `Revenue` data.")
        st.markdown("- Perform **Pareto Analysis** (Concentration Risk).")
        st.markdown("- Check Price Elasticity (Price vs Volume).")
        
        if st.button("Launch Q2: Product Strategy", use_container_width=True):
            st.switch_page("pages/2_Q2_Product_Analysis.py")

# ROW 2: Q3 & Q4
c3, c4 = st.columns(2)

with c3:
    with st.container(border=True):
        st.markdown("### 3️ Clinical Risk (CareTrack)")
        st.markdown("**Focus:** Correlation, Clustering (K-Means), Triage")
        st.error("Stratify patients by risk, find hidden cohorts using ML, and map vitals.")
        
        st.markdown("**Use this if you want to:**")
        st.markdown("- Correlate `BMI` vs `Disease Score`.")
        st.markdown("- Run **Unsupervised K-Means Clustering**.")
        st.markdown("- Auto-classify patients as High/Medium/Low risk.")
        
        if st.button("Launch Q3: Risk Analysis", use_container_width=True):
            st.switch_page("pages/3_Q3_Patient_Risk_Analysis.py")

with c4:
    with st.container(border=True):
        st.markdown("### 4️ Health Data Audit (EC Dept)")
        st.markdown("**Focus:** Ethics, Sampling (Monte Carlo), Bias Audit")
        st.warning("Audit data quality, detect algorithmic bias, and ensure ethical compliance.")
        
        st.markdown("**Use this if you want to:**")
        st.markdown("- Perform a **Monte Carlo** validation sample.")
        st.markdown("- Detect **Algorithmic Bias** in age groups.")
        st.markdown("- Check data hygiene (duplicates/missing).")
        
        if st.button("Launch Q4: Health Audit", use_container_width=True):
            st.switch_page("pages/4_Q4_Health_Data_Audit.py")

st.markdown("---")

# --- INTEGRATION FEATURE (REQUESTED) ---
with st.expander(" System Integration Status & Future Roadmap", expanded=False):
    st.markdown("""
    **Current Integration Capability:**
    The system currently operates as 4 distinct micro-services. To integrate them for a holistic view:
    
    1.  **Cross-Pollination:** Use the *Patient Risk Segments* from **Q3** to inform *Audit Sampling* in **Q4**.
    2.  **Revenue vs. Customer:** Export **Q1** customer trends and overlay them with **Q2** revenue peaks to see if new users drive high-value product sales.
    
    **Upcoming Features:**
    -  **Unified Data Lake:** Merge `customers.csv` and `products.csv` using `Transaction_ID`.
    -  **Meta-Model:** An AI agent that reads outputs from all 4 modules to generate a CEO-level executive summary.
    """)

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Eduvos Project Toolkit")
st.sidebar.caption("Student: ")
st.sidebar.caption("Module: ITPFA0-44")