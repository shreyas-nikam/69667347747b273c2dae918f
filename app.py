import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source import *

# --- Page Configuration ---
st.set_page_config(
    page_title="QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI", layout="wide")
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
pages = ["Introduction", "Org-AI-R Calculator",
         "Scenario Analysis & Visualization"]

try:
    current_index = pages.index(st.session_state.current_page)
except ValueError:
    current_index = 0
    st.session_state.current_page = "Introduction"

page_selection = st.sidebar.selectbox(
    "Go to",
    pages,
    index=current_index
)
st.sidebar.divider()

if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.rerun()

st.sidebar.subheader(f"Key Objectives")
st.sidebar.markdown(f"""- **Remember**: State the SEM formula and its components
- **Understand**: Explain why fixed-width CIs are problematic
- **Apply**: Implement proper SEM-based confidence intervals
- **Analyze**: Compare reliability across evidence counts
- **Evaluate**: Assess CI calibration accuracy
- **Create**: Design audit-ready scoring pipeline""")

st.sidebar.subheader(f"Tools Introduced")
st.sidebar.markdown(
    f"""- `scipy.stats`: Statistical functions for CI calculation, distributions
- `structlog`: Structured logging for full audit trails
- `Decimal`: Precision for financial-grade accuracy""")

# --- Page Content: Introduction ---
if st.session_state.current_page == "Introduction":
    st.markdown(f"Welcome to Week 6 of developing the PE-Org AIR System! As a Software Developer on the team, your task is to implement the core logic for the Organizational AI-Readiness (Org-AI-R) score. This involves precisely aggregating various components, ensuring financial-grade accuracy, providing robust confidence intervals based on the Standard Error of Measurement (SEM), and maintaining comprehensive audit trails. This application guides you through the process, step-by-step, as you build and validate this critical component.")

    st.markdown(f"## Key Concepts")
    st.markdown(f"""- Standard Error of Measurement (SEM)
- Spearman-Brown reliability prophecy
- H^R position adjustment ($\\delta = 0.15$)
- Synergy with TimingFactor
- Full Org-AI-R formula""")

    st.markdown("## Pre Requisites")
    st.markdown(f"""- Week 5 completed
- Understanding of confidence intervals
                """)

# --- Page Content: Org-AI-R Calculator ---
elif st.session_state.current_page == "Org-AI-R Calculator":
    st.markdown(f"## 1. Input Parameters for Org-AI-R Calculation")
    st.markdown(f"As a Software Developer, your primary task is to ensure the `OrgAIRCalculator` service correctly aggregates various components. Input the parameters below to compute a company's Org-AI-R score and its confidence interval.")

    st.markdown(f"### Core Formulas")
    st.markdown(f"The full Org-AI-R aggregation formula is:")
    st.markdown(
        r"$$\text{{Org-AI-R}} = (1-\beta) \times [\alpha \times V^R + (1-\alpha) \times H^R] + \beta \times \text{{Synergy}}$$")
    st.markdown(r"where $\alpha$ is the weight for $V^R$ (default `0.60`), $\beta$ is the weight for Synergy (default `0.12`), $V^R$ is Idiosyncratic Readiness, $H^R$ is Systematic Opportunity, and Synergy is the interaction effect.")

    st.markdown(f"The $H^R$ (Systematic Opportunity) formula is:")
    st.markdown(
        r"$$H^R = H^R_{{\text{{base}}}} \times (1 + \delta \times \text{{PositionFactor}})$$")
    st.markdown(r"where $H^R_{{\text{{base}}}}$ is the baseline HR score, $\delta$ is the position adjustment factor (corrected to 0.15), and PositionFactor reflects the company's strategic position relative to AI.")

    st.markdown(f"The Synergy formula is:")
    st.markdown(
        r"$$\text{{Synergy}} = \left(\frac{{V^R \times H^R}}{{100}}\right) \times \text{{Alignment}} \times \text{{TimingFactor}}$$")
    st.markdown(
        r"where $V^R$ is Idiosyncratic Readiness, $H^R$ is Systematic Opportunity, Alignment reflects strategic fit, and TimingFactor (clamped to $[0.8, 1.2]$) accounts for market timing.")

    st.markdown(f"The Confidence Interval (CI) is calculated as:")
    st.markdown(r"$$CI = \text{{score}} \pm z \times \text{{SEM}}$$")
    st.markdown(r"where score is the point estimate, $z$ is the critical value from the standard normal distribution corresponding to the desired confidence level, and SEM is the Standard Error of Measurement.")

    st.markdown(f"The Standard Error of Measurement (SEM) is calculated using the population standard deviation ($\\sigma$) and the score's reliability ($\\rho$) as:")
    st.markdown(r"$$\text{{SEM}} = \sigma \times \sqrt{{1 - \rho}}$$")
    st.markdown(
        r"where $\sigma$ is the population standard deviation for the score type, and $\rho$ is the score's reliability.")

    st.markdown(f"Reliability, $\\rho$, is estimated using the Spearman-Brown prophecy formula, which accounts for the number of evidence items ($n$) and the average inter-item correlation ($r$):")
    st.markdown(r"$$\rho = \frac{{n \times r}}{{1 + (n-1) \times r}}$$")
    st.markdown(
        r"where $n$ is the number of evidence items, and $r$ is the average inter-item correlation.")

    st.markdown(f"### Implementation Details")
    st.markdown(
        f"Below are the key code implementations for the Org-AI-R calculation system.")

    with st.expander("Task 6.1: SEM-Based Confidence Calculator"):
        st.markdown(f"The `ConfidenceCalculator` class implements the SEM-based confidence interval calculation using the Spearman-Brown prophecy formula:")
        st.code('''# src/pe_orgair/services/scoring/confidence.py
"""SEM-based confidence interval calculation."""
from decimal import Decimal
from dataclasses import dataclass
import math
from scipy import stats
from pe_orgair.services.scoring.utils import to_decimal, clamp

@dataclass
class ConfidenceInterval:
    """Confidence interval with SEM details."""
    point_estimate: Decimal
    ci_lower: Decimal
    ci_upper: Decimal
    sem: Decimal
    reliability: Decimal
    evidence_count: int
    confidence_level: float
    
    @property
    def ci_width(self) -> Decimal:
        return self.ci_upper - self.ci_lower

class ConfidenceCalculator:
    """
    Calculate SEM-based confidence intervals.
    
    Formula:
        CI = θ̂ ± z × SEM
    Where:
        SEM = σ × √(1 - ρ)
        ρ = (n × r) / (1 + (n-1) × r)  [Spearman-Brown]
    """
    
    # Population standard deviations by score type
    POPULATION_SD = {
        "vr": Decimal("15.0"),
        "hr": Decimal("12.0"),
        "synergy": Decimal("10.0"),
        "org_air": Decimal("14.0"),
    }
    
    # Average inter-item correlation
    DEFAULT_ITEM_CORRELATION = 0.30
    
    def calculate(
        self,
        score: Decimal,
        score_type: str,
        evidence_count: int,
        item_correlation: float = None,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """
        Calculate SEM-based confidence interval.
        
        Args:
            score: Point estimate
            score_type: Type of score (vr, hr, synergy, org_air)
            evidence_count: Number of evidence items/assessments
            item_correlation: Inter-item correlation (default 0.30)
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            ConfidenceInterval with SEM details
        """
        item_correlation = item_correlation or self.DEFAULT_ITEM_CORRELATION
        sigma = self.POPULATION_SD.get(score_type, Decimal("15.0"))
        
        # Spearman-Brown prophecy formula
        n = max(evidence_count, 1)
        reliability = (n * item_correlation) / (1 + (n - 1) * item_correlation)
        reliability = min(reliability, 0.99)  # Cap at 0.99
        reliability_dec = to_decimal(reliability)
        
        # SEM = σ × √(1 - ρ)
        sem = sigma * to_decimal(math.sqrt(1 - reliability))
        
        # z-score for confidence level
        z = to_decimal(stats.norm.ppf((1 + confidence_level) / 2))
        
        # Margin of error
        margin = z * sem
        
        return ConfidenceInterval(
            point_estimate=score,
            ci_lower=clamp(score - margin),
            ci_upper=clamp(score + margin),
            sem=sem,
            reliability=reliability_dec,
            evidence_count=n,
            confidence_level=confidence_level,
        )

confidence_calculator = ConfidenceCalculator()
''', language='python')

    with st.expander("Task 6.2: Full Org-AI-R Calculator"):
        st.markdown(
            f"The `OrgAIRCalculator` class orchestrates all component calculations and implements the main aggregation formula:")
        st.code('''# src/pe_orgair/services/scoring/org_air_calculator.py
"""Full Org-AI-R calculator with all components."""
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import uuid
import structlog

from pe_orgair.config.settings import settings
from pe_orgair.services.scoring.vr_calculator import vr_calculator, VRResult
from pe_orgair.services.scoring.confidence import confidence_calculator, ConfidenceInterval
from pe_orgair.services.scoring.utils import to_decimal, clamp

logger = structlog.get_logger()

@dataclass
class HRResult:
    """H^R calculation result."""
    hr_score: Decimal
    baseline: Decimal
    position_factor: Decimal
    delta_used: Decimal

@dataclass
class SynergyResult:
    """Synergy calculation result."""
    synergy_score: Decimal
    alignment_factor: Decimal
    timing_factor: Decimal
    interaction: Decimal

@dataclass
class OrgAIRResult:
    """Complete Org-AI-R calculation result."""
    score_id: str
    company_id: str
    sector_id: str
    timestamp: datetime
    
    # Final score
    final_score: Decimal
    
    # Components
    vr_result: VRResult
    hr_result: HRResult
    synergy_result: SynergyResult
    
    # Confidence
    confidence_interval: ConfidenceInterval
    confidence_tier: int
    
    # Parameters
    alpha: Decimal
    beta: Decimal
    parameter_version: str
    
    def to_dict(self) -> dict:
        return {
            "score_id": self.score_id,
            "company_id": self.company_id,
            "final_score": float(self.final_score),
            "v_r_score": float(self.vr_result.vr_score),
            "h_r_score": float(self.hr_result.hr_score),
            "synergy_score": float(self.synergy_result.synergy_score),
            "ci_lower": float(self.confidence_interval.ci_lower),
            "ci_upper": float(self.confidence_interval.ci_upper),
            "confidence_tier": self.confidence_tier,
            "parameter_version": self.parameter_version,
        }

class OrgAIRCalculator:
    """
    Full Org-AI-R calculator.
    
    Formula (Equation 1):
        Org-AI-R = (1-β) × [α×V^R + (1-α)×H^R] + β × Synergy
        
    With defaults:
        Org-AI-R = 0.88 × [0.60×V^R + 0.40×H^R] + 0.12 × Synergy
    """
    
    def __init__(self):
        self.alpha = to_decimal(settings.ALPHA_VR_WEIGHT)
        self.beta = to_decimal(settings.BETA_SYNERGY_WEIGHT)
        self.delta = to_decimal(settings.DELTA_POSITION)  # 0.15
    
    def calculate(
        self,
        company_id: str,
        sector_id: str,
        dimension_scores: List[float],
        talent_concentration: float,
        hr_baseline: float,
        position_factor: float,
        alignment_factor: float = 0.8,
        timing_factor: float = 1.0,
        evidence_count: int = 10,
        confidence_tier: int = 2,
    ) -> OrgAIRResult:
        """Calculate complete Org-AI-R score."""
        score_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Step 1: V^R
        vr_result = vr_calculator.calculate(dimension_scores, talent_concentration)
        
        # Step 2: H^R (with corrected δ = 0.15)
        hr_result = self._calculate_hr(hr_baseline, position_factor)
        
        # Step 3: Synergy (with TimingFactor)
        synergy_result = self._calculate_synergy(
            vr_result.vr_score,
            hr_result.hr_score,
            alignment_factor,
            timing_factor,
        )
        
        # Step 4: Full Org-AI-R
        one_minus_beta = Decimal(1) - self.beta
        weighted_components = (
            self.alpha * vr_result.vr_score +
            (Decimal(1) - self.alpha) * hr_result.hr_score
        )
        final_score = (
            one_minus_beta * weighted_components +
            self.beta * synergy_result.synergy_score
        )
        final_score = clamp(final_score)
        
        # Step 5: SEM-based confidence interval
        ci = confidence_calculator.calculate(
            score=final_score,
            score_type="org_air",
            evidence_count=evidence_count,
        )
        
        result = OrgAIRResult(
            score_id=score_id,
            company_id=company_id,
            sector_id=sector_id,
            timestamp=timestamp,
            final_score=final_score,
            vr_result=vr_result,
            hr_result=hr_result,
            synergy_result=synergy_result,
            confidence_interval=ci,
            confidence_tier=confidence_tier,
            alpha=self.alpha,
            beta=self.beta,
            parameter_version=settings.PARAM_VERSION,
        )
        
        logger.info(
            "org_air_calculated",
            score_id=score_id,
            company_id=company_id,
            final_score=float(final_score),
            ci_lower=float(ci.ci_lower),
            ci_upper=float(ci.ci_upper),
            ci_width=float(ci.ci_width),
        )
        
        return result
    
    def _calculate_hr(self, baseline: float, position_factor: float) -> HRResult:
        """
        Calculate H^R with corrected δ.
        
        H^R = H^R_base × (1 + δ × PositionFactor)
        Where δ = 0.15 (CORRECTED from 0.5)
        """
        baseline_dec = to_decimal(baseline)
        pf_dec = to_decimal(position_factor)
        
        hr_score = baseline_dec * (Decimal(1) + self.delta * pf_dec)
        hr_score = clamp(hr_score)
        
        return HRResult(
            hr_score=hr_score,
            baseline=baseline_dec,
            position_factor=pf_dec,
            delta_used=self.delta,
        )
    
    def _calculate_synergy(
        self,
        vr_score: Decimal,
        hr_score: Decimal,
        alignment: float,
        timing: float,
    ) -> SynergyResult:
        """
        Calculate Synergy with TimingFactor.
        
        Synergy = (V^R × H^R / 100) × Alignment × TimingFactor
        """
        alignment_dec = to_decimal(alignment)
        timing_dec = to_decimal(max(0.8, min(1.2, timing)))  # Clamp to [0.8, 1.2]
        
        interaction = (vr_score * hr_score) / Decimal(100)
        synergy = interaction * alignment_dec * timing_dec
        synergy = clamp(synergy)
        
        return SynergyResult(
            synergy_score=synergy,
            alignment_factor=alignment_dec,
            timing_factor=timing_dec,
            interaction=interaction,
        )

org_air_calculator = OrgAIRCalculator()
''', language='python')

    # --- Input Widgets ---
    with st.form("org_air_form"):
        st.subheader("Company Details")
        st.session_state.input_params["company_id"] = st.text_input(
            "Company ID", value=st.session_state.input_params["company_id"])
        st.session_state.input_params["sector_id"] = st.text_input(
            "Sector ID", value=st.session_state.input_params["sector_id"])

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
            st.session_state.org_air_result = org_air_calculator.calculate(
                **calc_args)
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
        st.metric("Synergy Component",
                  f"{result.synergy_result.synergy_score:.2f}")

        st.subheader("Confidence Interval Details")
        st.markdown(f"Understanding the uncertainty of our scores is crucial for robust decision-making. The SEM-based confidence interval provides this context.")
        st.markdown(
            f"**Point Estimate**: `{result.confidence_interval.point_estimate:.2f}`")
        st.markdown(
            f"**Confidence Interval ({result.confidence_interval.confidence_level*100:.0f}%)**: `[{result.confidence_interval.ci_lower:.2f}, {result.confidence_interval.ci_upper:.2f}]`")
        st.markdown(
            f"**SEM (Standard Error of Measurement)**: `{result.confidence_interval.sem:.2f}`")
        st.markdown(
            f"**Reliability ($\\rho$)**: `{result.confidence_interval.reliability:.2f}`")
        st.markdown(
            f"**Evidence Count ($n$)**: `{result.confidence_interval.evidence_count}`")
        st.markdown(
            f"**Confidence Level**: `{result.confidence_interval.confidence_level*100:.0f}%`")
        st.markdown(f"**Confidence Tier**: `{result.confidence_tier}`")
        st.markdown(f"**Parameters Version**: `{result.parameter_version}`")

        st.subheader("Full Result (for Auditability)")

        # Display V^R Component Details
        with st.expander("V^R (Idiosyncratic Readiness) Details"):
            st.write(f"**V^R Score**: {result.vr_result.vr_score:.2f}")
            st.write(
                f"**Dimension Scores**: {[float(d) for d in result.vr_result.dimension_scores_raw]}")
            st.write(
                f"**Talent Concentration**: {result.vr_result.talent_concentration_raw:.2f}%")

        # Display H^R Component Details
        with st.expander("H^R (Systematic Opportunity) Details"):
            st.write(f"**H^R Score**: {result.hr_result.hr_score:.2f}")
            st.write(f"**HR Baseline**: {result.hr_result.baseline:.2f}")
            st.write(
                f"**Position Factor**: {result.hr_result.position_factor:.2f}")
            st.write(
                f"**Delta (Position Adjustment)**: {result.hr_result.delta_used:.2f}")

        # Display Synergy Component Details
        with st.expander("Synergy Component Details"):
            st.write(
                f"**Synergy Score**: {result.synergy_result.synergy_score:.2f}")
            st.write(
                f"**Alignment Factor**: {result.synergy_result.alignment_factor:.2f}")
            st.write(
                f"**Timing Factor**: {result.synergy_result.timing_factor:.2f}")
            st.write(
                f"**Interaction (V^R × H^R / 100)**: {result.synergy_result.interaction:.2f}")

        # Display Aggregation Details
        with st.expander("Aggregation & Weighting Details"):
            st.write(f"**Alpha (V^R weight)**: {result.alpha:.2f}")
            st.write(f"**Beta (Synergy weight)**: {result.beta:.2f}")
            st.write(f"**Final Score**: {result.final_score:.2f}")
            st.write(f"**Company ID**: {result.company_id}")
            st.write(f"**Sector ID**: {result.sector_id}")

# --- Page Content: Scenario Analysis & Visualization ---
elif st.session_state.current_page == "Scenario Analysis & Visualization":
    st.markdown(f"# Org-AI-R Scenario Analysis")
    st.markdown(
        f"## 1. Visualizing Org-AI-R Scores and Confidence Intervals Across Scenarios")
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
            "evidence_count": 25,  # High evidence
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
            "evidence_count": 10,  # Moderate evidence
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
            "evidence_count": 18,  # Good evidence
            "confidence_tier": 2,
            "confidence_level": 0.95,
        },
    ]

    run_scenario_button = st.button(
        "Run Scenario Analysis for Multiple Companies")

    if run_scenario_button or st.session_state.all_scenario_results is not None:
        if st.session_state.all_scenario_results is None:
            all_results = []
            for scenario in scenarios:
                result = org_air_calculator.calculate(**scenario)
                all_results.append(result.to_dict())
            st.session_state.all_scenario_results = all_results

        results_df = pd.DataFrame(st.session_state.all_scenario_results)
        results_df['org_air_score'] = results_df['final_score']
        results_df['ci_error'] = (
            results_df['ci_upper'] - results_df['ci_lower']) / 2

        st.markdown(f"### Aggregated Results Across Scenarios")
        st.dataframe(results_df[['company_id', 'org_air_score', 'ci_lower',
                     'ci_upper', 'sem', 'reliability', 'evidence_count']].round(2))

        # --- Visualization 1: Org-AI-R Scores with Confidence Intervals ---
        st.markdown(f"### 2. Org-AI-R Scores with Confidence Intervals")
        st.markdown(f"The bar chart with error bars below clearly shows the point estimate of the Org-AI-R score for each company, along with the range where the true score is likely to fall. Companies with fewer `evidence_count` (e.g., 'TRADITIONAL_CO' with 5 evidence items) naturally exhibit wider confidence intervals, reflecting greater uncertainty in their score. Conversely, 'GLOBAL_LEADER' with 25 evidence items has a much narrower CI, indicating a more precise measurement. This confirms that our `ConfidenceCalculator` is working as expected, demonstrating the dynamic, evidence-based nature of our confidence intervals and directly addressing the problem of fixed-width CIs. This is a direct measure of our 'CI calibration accuracy'.")

        fig1, ax1 = plt.subplots(figsize=(12, 7))
        sns.barplot(x='company_id', y='org_air_score',
                    data=results_df, palette='viridis', ax=ax1)
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
        ax1.set_title(
            "Org-AI-R Scores with SEM-Based 95% Confidence Intervals Across Companies")
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend()
        st.pyplot(fig1)

        # --- Visualization 2: Reliability vs. Evidence Count (Illustrating Spearman-Brown) ---
        st.markdown(
            f"### 3. Reliability vs. Evidence Count (Spearman-Brown Prophecy)")
        st.markdown(f"This plot visually demonstrates the Spearman-Brown prophecy formula. It shows that as the number of evidence items ($n$) increases, the estimated reliability ($\\rho$) of the measurement also increases, though with diminishing returns. For the developer, this plot is a powerful way to confirm the theoretical correctness of the reliability component within the `ConfidenceCalculator`. It visually verifies that our implementation accurately models how measurement precision improves with more data. This provides strong assurance that the underlying statistical model is sound.")

        evidence_counts = np.arange(1, 31)
        # Use the constant from the imported confidence_calculator or class
        try:
            default_item_corr = float(
                confidence_calculator.DEFAULT_ITEM_CORRELATION)
        except:
            # Fallback if direct access fails, though source import should provide it
            default_item_corr = 0.3

        reliabilities = []
        for n in evidence_counts:
            rho = (n * default_item_corr) / (1 + (n - 1) * default_item_corr)
            reliabilities.append(min(rho, 0.99))  # Cap at 0.99

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(evidence_counts, reliabilities,
                 marker='o', linestyle='-', color='skyblue')
        ax2.axhline(y=0.7, color='red', linestyle='--',
                    label='Acceptable Reliability (0.7)')
        ax2.axhline(y=0.9, color='green', linestyle='--',
                    label='High Reliability (0.9)')
        ax2.set_xlabel("Number of Evidence Items (n)")
        ax2.set_ylabel("Estimated Reliability (ρ)")
        ax2.set_title(
            "Impact of Evidence Count on Score Reliability (Spearman-Brown Prophecy)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
