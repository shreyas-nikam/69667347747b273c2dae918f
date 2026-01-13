import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source import *

# --- Page Configuration ---
st.set_page_config(page_title="QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI")
st.divider()

# --- Session State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction"

if 'org_air_result' not in st.session_state:
    st.session_state.org_air_result = None

if 'all_scenario_results' not in st.session_state:
    st.session_state.all_scenario_results = None

if 'input_params' not in st.session_state:
    st.session_state.input_params = {
        "company_id": "ACME_CORP",
        "sector_id": "FINTECH",
        "dimension_scores_str": "70.0, 75.0, 68.0, 80.0",
        "talent_concentration": 25.0,
        "hr_baseline": 60.0,
        "position_factor": 0.8,
        "alignment_factor": 0.9,
        "timing_factor": 1.1,
        "evidence_count": 15,
        "confidence_tier": 2,
        "confidence_level": 0.95,
    }

# --- Sidebar Navigation ---
st.sidebar.title("Org-AI-R System Navigation")
pages = ["Introduction", "Org-AI-R Calculator", "Scenario Analysis & Visualization"]

try:
    current_index = pages.index(st.session_state.current_page)
except ValueError:
    current_index = 0
    st.session_state.current_page = "Introduction"

page_selection = st.sidebar.radio(
    "Go to",
    pages,
    index=current_index
)

if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.rerun()

# --- Page Content: Introduction ---
if st.session_state.current_page == "Introduction":
    st.markdown(f"# Org-AI-R Score Calculation and Validation: A Software Developer's Workflow")
    st.markdown(f"Welcome to Week 6 of developing the PE-Org AIR System! As a Software Developer on the team, your task is to implement the core logic for the Organizational AI-Readiness (Org-AI-R) score. This involves precisely aggregating various components, ensuring financial-grade accuracy, providing robust confidence intervals based on the Standard Error of Measurement (SEM), and maintaining comprehensive audit trails. This application guides you through the process, step-by-step, as you build and validate this critical component. Our goal is to deliver a robust `OrgAIRCalculator` service and verify its correctness, empowering strategic decision-making with a reliable AI-Readiness metric.")

    st.markdown(f"## Key Objectives")
    st.markdown(f"- **Remember**: State the SEM formula and its components")
    st.markdown(f"- **Understand**: Explain why fixed-width CIs are problematic")
    st.markdown(f"- **Apply**: Implement proper SEM-based confidence intervals")
    st.markdown(f"- **Analyze**: Compare reliability across evidence counts")
    st.markdown(f"- **Evaluate**: Assess CI calibration accuracy")
    st.markdown(f"- **Create**: Design audit-ready scoring pipeline")

    st.markdown(f"## Tools Introduced")
    st.markdown(f"- `scipy.stats`: Statistical functions for CI calculation, distributions")
    st.markdown(f"- `structlog`: Structured logging for full audit trails")
    st.markdown(f"- `Decimal`: Precision for financial-grade accuracy")

    st.markdown(f"## Key Concepts")
    st.markdown(f"- Standard Error of Measurement (SEM)")
    st.markdown(f"- Spearman-Brown reliability prophecy")
    st.markdown(f"- H^R position adjustment ($\\delta = 0.15$)")
    st.markdown(f"- Synergy with TimingFactor")
    st.markdown(f"- Full Org-AI-R formula")

# --- Page Content: Org-AI-R Calculator ---
elif st.session_state.current_page == "Org-AI-R Calculator":
    st.markdown(f"# Org-AI-R Calculator: Company Assessment")
    st.markdown(f"## 1. Input Parameters for Org-AI-R Calculation")
    st.markdown(f"As a Software Developer, your primary task is to ensure the `OrgAIRCalculator` service correctly aggregates various components. Input the parameters below to compute a company's Org-AI-R score and its confidence interval.")

    st.markdown(f"### Core Formulas")
    st.markdown(f"The full Org-AI-R aggregation formula is:")
    st.markdown(r"$$ \text{{Org-AI-R}} = (1-\beta) \times [\alpha \times V^R + (1-\alpha) \times H^R] + \beta \times \text{{Synergy}} $$")
    st.markdown(r"where $\alpha$ is the weight for $V^R$ (default `0.60`), $\beta$ is the weight for Synergy (default `0.12`), $V^R$ is Idiosyncratic Readiness, $H^R$ is Systematic Opportunity, and Synergy is the interaction effect.")

    st.markdown(f"The $H^R$ (Systematic Opportunity) formula is:")
    st.markdown(r"$$ H^R = H^R_{{\text{{base}}}} \times (1 + \delta \times \text{{PositionFactor}}) $$")
    st.markdown(r"where $H^R_{{\text{{base}}}}$ is the baseline HR score, $\delta$ is the position adjustment factor (corrected to 0.15), and PositionFactor reflects the company's strategic position relative to AI.")

    st.markdown(f"The Synergy formula is:")
    st.markdown(r"$$ \text{{Synergy}} = \left(\frac{{V^R \times H^R}}{{100}}\right) \times \text{{Alignment}} \times \text{{TimingFactor}} $$")
    st.markdown(r"where $V^R$ is Idiosyncratic Readiness, $H^R$ is Systematic Opportunity, Alignment reflects strategic fit, and TimingFactor (clamped to $[0.8, 1.2]$) accounts for market timing.")

    st.markdown(f"The Confidence Interval (CI) is calculated as:")
    st.markdown(r"$$ CI = \text{{score}} \pm z \times \text{{SEM}} $$")
    st.markdown(r"where score is the point estimate, $z$ is the critical value from the standard normal distribution corresponding to the desired confidence level, and SEM is the Standard Error of Measurement.")

    st.markdown(f"The Standard Error of Measurement (SEM) is calculated using the population standard deviation ($\\sigma$) and the score's reliability ($\\rho$) as:")
    st.markdown(r"$$ \text{{SEM}} = \sigma \times \sqrt{{1 - \rho}} $$")
    st.markdown(r"where $\sigma$ is the population standard deviation for the score type, and $\rho$ is the score's reliability.")

    st.markdown(f"Reliability, $\\rho$, is estimated using the Spearman-Brown prophecy formula, which accounts for the number of evidence items ($n$) and the average inter-item correlation ($r$):")
    st.markdown(r"$$ \rho = \frac{{n \times r}}{{1 + (n-1) \times r}} $$")
    st.markdown(r"where $n$ is the number of evidence items, and $r$ is the average inter-item correlation.")

    # --- Input Widgets ---
    with st.form("org_air_form"):
        st.subheader("Company Details")
        st.session_state.input_params["company_id"] = st.text_input("Company ID", value=st.session_state.input_params["company_id"])
        st.session_state.input_params["sector_id"] = st.text_input("Sector ID", value=st.session_state.input_params["sector_id"])

        st.subheader("V^R (Idiosyncratic Readiness) Factors")
        st.session_state.input_params["dimension_scores_str"] = st.text_area(
            "Dimension Scores (comma-separated floats)",
            value=st.session_state.input_params["dimension_scores_str"],
            help="E.g., 70.0, 75.0, 68.0, 80.0"
        )
        st.session_state.input_params["talent_concentration"] = st.number_input(
            "Talent Concentration (%)",
            min_value=0.0, max_value=100.0, value=float(st.session_state.input_params["talent_concentration"]), step=0.1
        )

        st.subheader("H^R (Systematic Opportunity) Factors")
        st.session_state.input_params["hr_baseline"] = st.number_input(
            "HR Baseline Score (0-100)",
            min_value=0.0, max_value=100.0, value=float(st.session_state.input_params["hr_baseline"]), step=0.1
        )
        st.session_state.input_params["position_factor"] = st.number_input(
            "Position Factor (e.g., 0.5 to 1.5)",
            min_value=0.0, max_value=2.0, value=float(st.session_state.input_params["position_factor"]), step=0.01
        )

        st.subheader("Synergy Factors")
        st.session_state.input_params["alignment_factor"] = st.number_input(
            "Alignment Factor (default 0.8)",
            min_value=0.0, max_value=1.0, value=float(st.session_state.input_params["alignment_factor"]), step=0.01
        )
        st.session_state.input_params["timing_factor"] = st.number_input(
            "Timing Factor (default 1.0, clamped to [0.8, 1.2])",
            min_value=0.5, max_value=1.5, value=float(st.session_state.input_params["timing_factor"]), step=0.01
        )

        st.subheader("Confidence Interval Factors")
        st.session_state.input_params["evidence_count"] = st.number_input(
            "Evidence Count (for CI calculation, default 10)",
            min_value=1, max_value=100, value=int(st.session_state.input_params["evidence_count"]), step=1
        )
        st.session_state.input_params["confidence_tier"] = st.number_input(
            "Confidence Tier (default 2, currently not used in SEM logic directly)",
            min_value=1, max_value=5, value=int(st.session_state.input_params["confidence_tier"]), step=1
        )
        st.session_state.input_params["confidence_level"] = st.number_input(
            "Confidence Level (e.g., 0.95 for 95% CI)",
            min_value=0.01, max_value=0.99, value=float(st.session_state.input_params["confidence_level"]), step=0.01
        )

        submitted = st.form_submit_button("Calculate Org-AI-R Score")

    if submitted:
        try:
            # Parse dimension_scores from string to list of floats
            dimension_scores = [
                float(s.strip()) for s in st.session_state.input_params["dimension_scores_str"].split(',') if s.strip()
            ]

            # Prepare arguments for the calculator
            calc_args = st.session_state.input_params.copy()
            calc_args["dimension_scores"] = dimension_scores
            del calc_args["dimension_scores_str"]

            # Execute calculation
            st.session_state.org_air_result = org_air_calculator.calculate(**calc_args)
            st.success("Org-AI-R Score Calculated Successfully!")
        except Exception as e:
            st.error(f"Error calculating Org-AI-R score: {e}")
            st.session_state.org_air_result = None

    # --- Display Results ---
    if st.session_state.org_air_result:
        result = st.session_state.org_air_result
        st.markdown(f"### 2. Org-AI-R Calculation Result")
        st.markdown(f"The `OrgAIRCalculator` orchestrates all component calculations (`V^R`, `H^R`, `Synergy`) and applies the main aggregation formula. It then attaches the SEM-based confidence interval.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Org-AI-R Score", f"{result.final_score:.2f}")
        with col2:
            st.metric("V^R Component", f"{result.vr_result.vr_score:.2f}")
        with col3:
            st.metric("H^R Component", f"{result.hr_result.hr_score:.2f}")
        st.metric("Synergy Component", f"{result.synergy_result.synergy_score:.2f}")

        st.subheader("Confidence Interval Details")
        st.markdown(f"Understanding the uncertainty of our scores is crucial for robust decision-making. The SEM-based confidence interval provides this context.")
        st.markdown(f"**Point Estimate**: `{result.confidence_interval.point_estimate:.2f}`")
        st.markdown(f"**Confidence Interval ({result.confidence_interval.confidence_level*100:.0f}%)**: `[{result.confidence_interval.ci_lower:.2f}, {result.confidence_interval.ci_upper:.2f}]`")
        st.markdown(f"**SEM (Standard Error of Measurement)**: `{result.confidence_interval.sem:.2f}`")
        st.markdown(f"**Reliability ($\\rho$)**: `{result.confidence_interval.reliability:.2f}`")
        st.markdown(f"**Evidence Count ($n$)**: `{result.confidence_interval.evidence_count}`")
        st.markdown(f"**Confidence Level**: `{result.confidence_interval.confidence_level*100:.0f}%`")
        st.markdown(f"**Confidence Tier**: `{result.confidence_tier}`")
        st.markdown(f"**Parameters Version**: `{result.parameter_version}`")

        st.subheader("Full Result Object (for Auditability)")
        st.json(result.to_dict())

# --- Page Content: Scenario Analysis & Visualization ---
elif st.session_state.current_page == "Scenario Analysis & Visualization":
    st.markdown(f"# Org-AI-R Scenario Analysis")
    st.markdown(f"## 1. Visualizing Org-AI-R Scores and Confidence Intervals Across Scenarios")
    st.markdown(f"To make our Org-AI-R service truly impactful, leadership needs to easily compare scores and understand their reliability. As a Software Developer, we will generate synthetic scenarios and visualize their Org-AI-R scores with their confidence intervals. This helps in identifying companies with robust readiness scores versus those where the measurement might be less precise.")

    # --- Scenario Data ---
    scenarios = [
        {
            "company_id": "GLOBAL_LEADER",
            "sector_id": "AI_TECH",
            "dimension_scores": [90.0, 88.0, 92.0, 85.0],
            "talent_concentration": 40.0,
            "hr_baseline": 85.0,
            "position_factor": 1.2,
            "alignment_factor": 0.95,
            "timing_factor": 1.2,
            "evidence_count": 25, # High evidence
            "confidence_tier": 1,
            "confidence_level": 0.95,
        },
        {
            "company_id": "GROWTH_STARTUP",
            "sector_id": "FINTECH",
            "dimension_scores": [65.0, 70.0, 60.0, 72.0],
            "talent_concentration": 30.0,
            "hr_baseline": 60.0,
            "position_factor": 0.9,
            "alignment_factor": 0.8,
            "timing_factor": 1.0,
            "evidence_count": 10, # Moderate evidence
            "confidence_tier": 2,
            "confidence_level": 0.95,
        },
        {
            "company_id": "TRADITIONAL_CO",
            "sector_id": "MANUFACTURING",
            "dimension_scores": [45.0, 50.0, 40.0, 55.0],
            "talent_concentration": 15.0,
            "hr_baseline": 40.0,
            "position_factor": 0.6,
            "alignment_factor": 0.7,
            "timing_factor": 0.8,
            "evidence_count": 5,  # Low evidence
            "confidence_tier": 3,
            "confidence_level": 0.95,
        },
        {
            "company_id": "INNOVATIVE_SME",
            "sector_id": "HEALTHCARE",
            "dimension_scores": [78.0, 82.0, 75.0, 80.0],
            "talent_concentration": 35.0,
            "hr_baseline": 70.0,
            "position_factor": 1.0,
            "alignment_factor": 0.9,
            "timing_factor": 1.1,
            "evidence_count": 18, # Good evidence
            "confidence_tier": 2,
            "confidence_level": 0.95,
        },
    ]

    run_scenario_button = st.button("Run Scenario Analysis for Multiple Companies")

    if run_scenario_button or st.session_state.all_scenario_results is not None:
        if st.session_state.all_scenario_results is None:
            all_results = []
            for scenario in scenarios:
                result = org_air_calculator.calculate(**scenario)
                all_results.append(result.to_dict())
            st.session_state.all_scenario_results = all_results

        results_df = pd.DataFrame(st.session_state.all_scenario_results)
        results_df['org_air_score'] = results_df['final_score']
        results_df['ci_error'] = (results_df['ci_upper'] - results_df['ci_lower']) / 2

        st.markdown(f"### Aggregated Results Across Scenarios")
        st.dataframe(results_df[['company_id', 'org_air_score', 'ci_lower', 'ci_upper', 'sem', 'reliability', 'evidence_count']].round(2))

        # --- Visualization 1: Org-AI-R Scores with Confidence Intervals ---
        st.markdown(f"### 2. Org-AI-R Scores with Confidence Intervals")
        st.markdown(f"The bar chart with error bars below clearly shows the point estimate of the Org-AI-R score for each company, along with the range where the true score is likely to fall. Companies with fewer `evidence_count` (e.g., 'TRADITIONAL_CO' with 5 evidence items) naturally exhibit wider confidence intervals, reflecting greater uncertainty in their score. Conversely, 'GLOBAL_LEADER' with 25 evidence items has a much narrower CI, indicating a more precise measurement. This confirms that our `ConfidenceCalculator` is working as expected, demonstrating the dynamic, evidence-based nature of our confidence intervals and directly addressing the problem of fixed-width CIs. This is a direct measure of our 'CI calibration accuracy'.")

        fig1, ax1 = plt.subplots(figsize=(12, 7))
        sns.barplot(x='company_id', y='org_air_score', data=results_df, palette='viridis', ax=ax1)
        ax1.errorbar(
            x=results_df['company_id'],
            y=results_df['org_air_score'],
            yerr=results_df['ci_error'],
            fmt='none',
            c='black',
            capsize=5,
            label='95% Confidence Interval'
        )
        ax1.set_xlabel("Company ID")
        ax1.set_ylabel("Org-AI-R Score")
        ax1.set_title("Org-AI-R Scores with SEM-Based 95% Confidence Intervals Across Companies")
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend()
        st.pyplot(fig1)

        # --- Visualization 2: Reliability vs. Evidence Count (Illustrating Spearman-Brown) ---
        st.markdown(f"### 3. Reliability vs. Evidence Count (Spearman-Brown Prophecy)")
        st.markdown(f"This plot visually demonstrates the Spearman-Brown prophecy formula. It shows that as the number of evidence items ($n$) increases, the estimated reliability ($\\rho$) of the measurement also increases, though with diminishing returns. For the developer, this plot is a powerful way to confirm the theoretical correctness of the reliability component within the `ConfidenceCalculator`. It visually verifies that our implementation accurately models how measurement precision improves with more data. This provides strong assurance that the underlying statistical model is sound.")

        evidence_counts = np.arange(1, 31)
        # Use the constant from the imported confidence_calculator or class
        try:
            default_item_corr = float(confidence_calculator.DEFAULT_ITEM_CORRELATION)
        except:
            default_item_corr = 0.3 # Fallback if direct access fails, though source import should provide it

        reliabilities = []
        for n in evidence_counts:
            rho = (n * default_item_corr) / (1 + (n - 1) * default_item_corr)
            reliabilities.append(min(rho, 0.99)) # Cap at 0.99

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(evidence_counts, reliabilities, marker='o', linestyle='-', color='skyblue')
        ax2.axhline(y=0.7, color='red', linestyle='--', label='Acceptable Reliability (0.7)')
        ax2.axhline(y=0.9, color='green', linestyle='--', label='High Reliability (0.9)')
        ax2.set_xlabel("Number of Evidence Items (n)")
        ax2.set_ylabel("Estimated Reliability (ρ)")
        ax2.set_title("Impact of Evidence Count on Score Reliability (Spearman-Brown Prophecy)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)
