# app.py
# Refactored to use the function-driven org_air_core module (the “above” refactor).
# Assumes your refactored core code is saved as: org_air_core.py (or adjust imports accordingly)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from source import (
    build_calculators,
    results_to_records,
)

# --------------------------------------------------------------------------------------
# App Wiring (singleton-ish)
# --------------------------------------------------------------------------------------


@st.cache_resource
def get_bundle():
    # dev_console_logger=True for local; set False for JSON logs in prod
    return build_calculators(dev_console_logger=True, decimal_precision=10)


bundle = get_bundle()
org_air_calculator = bundle.org_air_calculator
# for DEFAULT_ITEM_CORRELATION access if needed
confidence_calculator = bundle.confidence_calculator

# --------------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI")
st.divider()

# --------------------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------------------

if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction"

if "org_air_result" not in st.session_state:
    st.session_state.org_air_result = None

if "all_scenario_results" not in st.session_state:
    st.session_state.all_scenario_results = None

if "input_params" not in st.session_state:
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

# --------------------------------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------------------------------

pages = ["Introduction", "Org-AI-R Calculator",
         "Scenario Analysis & Visualization"]

try:
    current_index = pages.index(st.session_state.current_page)
except ValueError:
    current_index = 0
    st.session_state.current_page = "Introduction"

page_selection = st.sidebar.selectbox("Go to", pages, index=current_index)
st.sidebar.divider()

if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.rerun()

st.sidebar.subheader("Key Objectives")
st.sidebar.markdown(
    """- **Remember**: State the SEM formula and its components
- **Understand**: Explain why fixed-width CIs are problematic
- **Apply**: Implement proper SEM-based confidence intervals
- **Analyze**: Compare reliability across evidence counts
- **Evaluate**: Assess CI calibration accuracy
- **Create**: Design audit-ready scoring pipeline"""
)

st.sidebar.subheader("Tools Introduced")
st.sidebar.markdown(
    """- `scipy.stats`: Statistical functions for CI calculation, distributions
- `structlog`: Structured logging for full audit trails
- `Decimal`: Precision for financial-grade accuracy"""
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def parse_dimension_scores(scores_str: str) -> list[float]:
    return [float(s.strip()) for s in scores_str.split(",") if s.strip()]


# --------------------------------------------------------------------------------------
# Page: Introduction
# --------------------------------------------------------------------------------------

if st.session_state.current_page == "Introduction":
    st.markdown(
        "Welcome to Week 6 of developing the PE-Org AIR System! As a Software Developer on the team, your task is to "
        "implement the core logic for the Organizational AI-Readiness (Org-AI-R) score. This involves precisely "
        "aggregating various components, ensuring financial-grade accuracy, providing robust confidence intervals based "
        "on the Standard Error of Measurement (SEM), and maintaining comprehensive audit trails."
    )

    st.markdown("## Key Concepts")
    st.markdown(
        """- Standard Error of Measurement (SEM)
- Spearman-Brown reliability prophecy
- H^R position adjustment ($\\delta = 0.15$)
- Synergy with TimingFactor
- Full Org-AI-R formula"""
    )

    st.markdown("## Pre Requisites")
    st.markdown(
        """- Week 5 completed
- Understanding of confidence intervals"""
    )

# --------------------------------------------------------------------------------------
# Page: Org-AI-R Calculator
# --------------------------------------------------------------------------------------

elif st.session_state.current_page == "Org-AI-R Calculator":
    st.markdown("## 1. Input Parameters for Org-AI-R Calculation")
    st.markdown(
        "As a Software Developer, your primary task is to ensure the OrgAIRCalculator service correctly aggregates "
        "various components. Input the parameters below to compute a company's Org-AI-R score and its confidence interval."
    )

    st.markdown("### Core Formulas")
    st.markdown(
        r"""$$
\text{{Org-AI-R}} = (1-\beta) \times [\alpha \times V^R + (1-\alpha) \times H^R] + \beta \times \text{{Synergy}}
$$"""
    )
    st.markdown(
        r"""$$
H^R = H^R_{{\text{{base}}}} \times (1 + \delta \times \text{{PositionFactor}})
$$"""
    )
    st.markdown(
        r"""$$
\text{{Synergy}} = \left(\frac{{V^R \times H^R}}{{100}}\right) \times \text{{Alignment}} \times \text{{TimingFactor}}
$$"""
    )
    st.markdown(r"""$$
CI = \text{{score}} \pm z \times \text{{SEM}}
$$""")
    st.markdown(r"""$$
\text{{SEM}} = \sigma \times \sqrt{{1 - \rho}}
$$""")
    st.markdown(r"""$$
\rho = \frac{{n \times r}}{{1 + (n-1) \times r}}
$$""")

    # --- Input Widgets ---
    with st.form("org_air_form"):
        st.subheader("Company Details")
        st.session_state.input_params["company_id"] = st.text_input(
            "Company ID", value=st.session_state.input_params["company_id"]
        )
        st.session_state.input_params["sector_id"] = st.text_input(
            "Sector ID", value=st.session_state.input_params["sector_id"]
        )

        st.subheader("V^R (Idiosyncratic Readiness) Factors")
        st.session_state.input_params["dimension_scores_str"] = st.text_area(
            "Dimension Scores (comma-separated floats)",
            value=st.session_state.input_params["dimension_scores_str"],
            help="E.g., 70.0, 75.0, 68.0, 80.0",
        )
        st.session_state.input_params["talent_concentration"] = st.number_input(
            "Talent Concentration (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.input_params["talent_concentration"]),
            step=0.1,
        )

        st.subheader("H^R (Systematic Opportunity) Factors")
        st.session_state.input_params["hr_baseline"] = st.number_input(
            "HR Baseline Score (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.input_params["hr_baseline"]),
            step=0.1,
        )
        st.session_state.input_params["position_factor"] = st.number_input(
            "Position Factor (e.g., 0.5 to 1.5)",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.input_params["position_factor"]),
            step=0.01,
        )

        st.subheader("Synergy Factors")
        st.session_state.input_params["alignment_factor"] = st.number_input(
            "Alignment Factor (default 0.8)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.input_params["alignment_factor"]),
            step=0.01,
        )
        st.session_state.input_params["timing_factor"] = st.number_input(
            "Timing Factor (default 1.0, clamped to [0.8, 1.2])",
            min_value=0.5,
            max_value=1.5,
            value=float(st.session_state.input_params["timing_factor"]),
            step=0.01,
        )

        st.subheader("Confidence Interval Factors")
        st.session_state.input_params["evidence_count"] = st.number_input(
            "Evidence Count (for CI calculation, default 10)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.input_params["evidence_count"]),
            step=1,
        )
        st.session_state.input_params["confidence_tier"] = st.number_input(
            "Confidence Tier (default 2, currently not used in SEM logic directly)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.input_params["confidence_tier"]),
            step=1,
        )
        st.session_state.input_params["confidence_level"] = st.number_input(
            "Confidence Level (e.g., 0.95 for 95% CI)",
            min_value=0.01,
            max_value=0.99,
            value=float(st.session_state.input_params["confidence_level"]),
            step=0.01,
        )

        submitted = st.form_submit_button("Calculate Org-AI-R Score")

    if submitted:
        try:
            dimension_scores = parse_dimension_scores(
                st.session_state.input_params["dimension_scores_str"])

            calc_args = st.session_state.input_params.copy()
            calc_args["dimension_scores"] = dimension_scores
            del calc_args["dimension_scores_str"]

            st.session_state.org_air_result = org_air_calculator.calculate(
                **calc_args)
            st.success("Org-AI-R Score Calculated Successfully!")
        except Exception as e:
            st.error(f"Error calculating Org-AI-R score: {e}")
            st.session_state.org_air_result = None

    # --- Display Results ---
    if st.session_state.org_air_result is not None:
        result = st.session_state.org_air_result

        st.markdown("### 2. Org-AI-R Calculation Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Org-AI-R Score", f"{result.final_score:.2f}")
        with col2:
            st.metric("V^R Component", f"{result.vr_result.vr_score:.2f}")
        with col3:
            st.metric("H^R Component", f"{result.hr_result.hr_score:.2f}")
        st.metric("Synergy Component",
                  f"{result.synergy_result.synergy_score:.2f}")

        st.subheader("Confidence Interval Details")
        st.markdown(
            f"**Point Estimate**: `{result.confidence_interval.point_estimate:.2f}`")
        st.markdown(
            f"**Confidence Interval ({result.confidence_interval.confidence_level*100:.0f}%)**: "
            f"`[{result.confidence_interval.ci_lower:.2f}, {result.confidence_interval.ci_upper:.2f}]`"
        )
        st.markdown(f"**SEM**: `{result.confidence_interval.sem:.2f}`")
        st.markdown(
            f"**Reliability ($\\rho$)**: `{result.confidence_interval.reliability:.2f}`")
        st.markdown(
            f"**Evidence Count ($n$)**: `{result.confidence_interval.evidence_count}`")
        st.markdown(f"**Confidence Tier**: `{result.confidence_tier}`")
        st.markdown(f"**Parameters Version**: `{result.parameter_version}`")

        st.subheader("Full Result (for Auditability)")

        with st.expander("V^R (Idiosyncratic Readiness) Details"):
            st.write(f"**V^R Score**: {result.vr_result.vr_score:.2f}")
            st.write(
                f"**Dimension Scores**: {[float(d) for d in result.vr_result.dimension_scores_raw]}")
            st.write(
                f"**Talent Concentration**: {result.vr_result.talent_concentration_raw:.2f}%")

        with st.expander("H^R (Systematic Opportunity) Details"):
            st.write(f"**H^R Score**: {result.hr_result.hr_score:.2f}")
            st.write(f"**HR Baseline**: {result.hr_result.baseline:.2f}")
            st.write(
                f"**Position Factor**: {result.hr_result.position_factor:.2f}")
            st.write(
                f"**Delta (Position Adjustment)**: {result.hr_result.delta_used:.2f}")

        with st.expander("Synergy Component Details"):
            st.write(
                f"**Synergy Score**: {result.synergy_result.synergy_score:.2f}")
            st.write(
                f"**Alignment Factor**: {result.synergy_result.alignment_factor:.2f}")
            st.write(
                f"**Timing Factor**: {result.synergy_result.timing_factor:.2f}")
            st.write(
                f"**Interaction (V^R × H^R / 100)**: {result.synergy_result.interaction:.2f}")

        with st.expander("Aggregation & Weighting Details"):
            st.write(f"**Alpha (V^R weight)**: {result.alpha:.2f}")
            st.write(f"**Beta (Synergy weight)**: {result.beta:.2f}")
            st.write(f"**Final Score**: {result.final_score:.2f}")
            st.write(f"**Company ID**: {result.company_id}")
            st.write(f"**Sector ID**: {result.sector_id}")

# --------------------------------------------------------------------------------------
# Page: Scenario Analysis & Visualization
# --------------------------------------------------------------------------------------

elif st.session_state.current_page == "Scenario Analysis & Visualization":
    st.markdown("# Org-AI-R Scenario Analysis")
    st.markdown(
        "## 1. Visualizing Org-AI-R Scores and Confidence Intervals Across Scenarios")

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
            "evidence_count": 25,
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
            "evidence_count": 10,
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
            "evidence_count": 5,
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
            "evidence_count": 18,
            "confidence_tier": 2,
            "confidence_level": 0.95,
        },
    ]

    run_scenario_button = st.button(
        "Run Scenario Analysis for Multiple Companies")

    if run_scenario_button or st.session_state.all_scenario_results is not None:
        if st.session_state.all_scenario_results is None:
            # ✅ now uses org_air_core outputs
            scenario_results = [
                org_air_calculator.calculate(**s) for s in scenarios]
            st.session_state.all_scenario_results = results_to_records(
                scenario_results)

        results_df = pd.DataFrame(st.session_state.all_scenario_results)
        results_df["org_air_score"] = results_df["final_score"]
        results_df["ci_error"] = (
            results_df["ci_upper"] - results_df["ci_lower"]) / 2

        st.markdown("### Aggregated Results Across Scenarios")
        st.dataframe(
            results_df[
                ["company_id", "org_air_score", "ci_lower",
                    "ci_upper", "sem", "reliability", "evidence_count"]
            ].round(2)
        )

        st.markdown("### 2. Org-AI-R Scores with Confidence Intervals")

        fig1, ax1 = plt.subplots(figsize=(12, 7))
        sns.barplot(x="company_id", y="org_air_score",
                    data=results_df, palette="viridis", ax=ax1)
        ax1.errorbar(
            x=results_df["company_id"],
            y=results_df["org_air_score"],
            yerr=results_df["ci_error"],
            fmt="none",
            c="black",
            capsize=5,
            label="95% Confidence Interval",
        )
        ax1.set_xlabel("Company ID")
        ax1.set_ylabel("Org-AI-R Score")
        ax1.set_title(
            "Org-AI-R Scores with SEM-Based 95% Confidence Intervals Across Companies")
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax1.legend()
        st.pyplot(fig1)

        st.markdown(
            "### 3. Reliability vs. Evidence Count (Spearman-Brown Prophecy)")

        evidence_counts = np.arange(1, 31)
        default_item_corr = float(
            confidence_calculator.DEFAULT_ITEM_CORRELATION)

        reliabilities = []
        for n in evidence_counts:
            rho = (n * default_item_corr) / (1 + (n - 1) * default_item_corr)
            reliabilities.append(min(rho, 0.99))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(evidence_counts, reliabilities,
                 marker="o", linestyle="-", color="skyblue")
        ax2.axhline(y=0.7, color="red", linestyle="--",
                    label="Acceptable Reliability (0.7)")
        ax2.axhline(y=0.9, color="green", linestyle="--",
                    label="High Reliability (0.9)")
        ax2.set_xlabel("Number of Evidence Items (n)")
        ax2.set_ylabel("Estimated Reliability (ρ)")
        ax2.set_title(
            "Impact of Evidence Count on Score Reliability (Spearman-Brown Prophecy)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)

# --------------------------------------------------------------------------------------
# License
# --------------------------------------------------------------------------------------

st.caption(
    """
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
"""
)
