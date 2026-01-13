from decimal import Decimal, getcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import uuid
import math

import structlog
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure Decimal precision for financial-grade accuracy
getcontext().prec = 10 # Set precision to 10 significant digits

# --- Utility Functions ---
def to_decimal(value) -> Decimal:
    """Converts a float or int to Decimal, handling None safely."""
    if value is None:
        return Decimal("0.0") # Or raise an error, depending on requirements
    return Decimal(str(value))

def clamp(value: Decimal, min_val: Decimal = Decimal("0.0"), max_val: Decimal = Decimal("100.0")) -> Decimal:
    """Clamps a Decimal value within a specified range."""
    return max(min_val, min(max_val, value))

# --- Mocking System Settings and VR Calculator ---
# In a real system, these would be loaded from a config file or service.
class Settings:
    ALPHA_VR_WEIGHT = 0.60 # Weight for V^R in Org-AI-R aggregation
    BETA_SYNERGY_WEIGHT = 0.12 # Weight for Synergy in Org-AI-R aggregation
    DELTA_POSITION = 0.15 # Corrected delta for H^R calculation
    PARAM_VERSION = "1.0.0" # Version of the parameters used

settings = Settings()

@dataclass
class VRResult:
    vr_score: Decimal
    dimension_scores_raw: List[Decimal]
    talent_concentration_raw: Decimal

class VRCalculator:
    """
    A placeholder for the V^R (Idiosyncratic Readiness) calculator.
    In a real system, this would involve complex logic based on multiple dimensions.
    For this lab, we simulate a simple aggregation.
    """
    def calculate(self, dimension_scores: List[float], talent_concentration: float) -> VRResult:
        dim_scores_dec = [to_decimal(s) for s in dimension_scores]
        talent_dec = to_decimal(talent_concentration)
        # Simple aggregation for demo: average of dimension scores, adjusted by talent concentration
        # Talent concentration acts as a multiplier factor, e.g., +1% for every 10% talent concentration
        if not dim_scores_dec:
            avg_dim_score = Decimal("0.0")
        else:
            avg_dim_score = sum(dim_scores_dec) / to_decimal(len(dim_scores_dec))

        # Talent concentration factor: 1 + (talent_concentration / 100)
        talent_factor = Decimal("1.0") + (talent_dec / Decimal("100.0"))

        vr_score = avg_dim_score * talent_factor
        vr_score = clamp(vr_score) # Ensure score is within 0-100 range
        return VRResult(vr_score=vr_score, dimension_scores_raw=dim_scores_dec, talent_concentration_raw=talent_dec)

vr_calculator = VRCalculator()

# --- Dataclasses for Results ---
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
    interaction: Decimal # This is (V^R * H^R / 100) before alignment and timing factors

@dataclass
class OrgAIRResult:
    """Complete Org-AI-R calculation result."""
    score_id: str
    company_id: str
    sector_id: str
    timestamp: datetime
    final_score: Decimal
    vr_result: VRResult
    hr_result: HRResult
    synergy_result: SynergyResult
    confidence_interval: ConfidenceInterval
    confidence_tier: int # Retained from original spec, not used in SEM calculation
    alpha: Decimal
    beta: Decimal
    parameter_version: str

    def to_dict(self) -> dict:
        """Converts the OrgAIRResult to a dictionary for logging/storage."""
        return {
            "score_id": self.score_id,
            "company_id": self.company_id,
            "sector_id": self.sector_id,
            "timestamp": self.timestamp.isoformat(),
            "final_score": float(self.final_score),
            "v_r_score": float(self.vr_result.vr_score),
            "h_r_score": float(self.hr_result.hr_score),
            "synergy_score": float(self.synergy_result.synergy_score),
            "ci_lower": float(self.confidence_interval.ci_lower),
            "ci_upper": float(self.confidence_interval.ci_upper),
            "ci_width": float(self.confidence_interval.ci_width),
            "sem": float(self.confidence_interval.sem),
            "reliability": float(self.confidence_interval.reliability),
            "evidence_count": self.confidence_interval.evidence_count,
            "confidence_level": self.confidence_interval.confidence_level,
            "confidence_tier": self.confidence_tier,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "parameter_version": self.parameter_version,
        }

# --- Structured Logger Setup ---
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(), # For console output, use JSONRenderer in production
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()
class ConfidenceCalculator:
    """Calculates SEM-based confidence intervals."""

    # Population standard deviations by score type (simulated)
    POPULATION_SD = {
        "vr": to_decimal("15.0"),
        "hr": to_decimal("12.0"),
        "synergy": to_decimal("10.0"),
        "org_air": to_decimal("14.0"),
    }
    # Average inter-item correlation (default)
    DEFAULT_ITEM_CORRELATION = to_decimal("0.30")

    def calculate(
        self,
        score: Decimal,
        score_type: str,
        evidence_count: int,
        item_correlation: Optional[float] = None,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """
        Calculate SEM-based confidence interval.

        Args:
            score: Point estimate for the score.
            score_type: Type of score (e.g., "vr", "hr", "synergy", "org_air").
            evidence_count: Number of evidence items/assessments used for the score.
            item_correlation: Inter-item correlation (default 0.30).
            confidence_level: Confidence level (default 0.95).

        Returns:
            ConfidenceInterval with SEM details.
        """
        item_correlation_dec = to_decimal(item_correlation) if item_correlation is not None else self.DEFAULT_ITEM_CORRELATION

        # Ensure evidence_count is at least 1 to avoid division by zero or negative reliability
        n = max(evidence_count, 1)

        # Spearman-Brown prophecy formula
        # ρ = (n * r) / (1 + (n-1) * r)
        # Cap reliability at 0.99 to avoid math domain errors with sqrt(1 - reliability) if reliability becomes 1.0
        reliability = (to_decimal(n) * item_correlation_dec) / (Decimal("1.0") + (to_decimal(n) - Decimal("1.0")) * item_correlation_dec)
        reliability = clamp(reliability, Decimal("0.0"), Decimal("0.99")) # Clamp reliability to a reasonable range

        # Population standard deviation for the given score type
        sigma = self.POPULATION_SD.get(score_type, to_decimal("15.0")) # Default sigma if score_type not found

        # SEM = σ × √(1 - ρ)
        sem = sigma * to_decimal(math.sqrt(float(Decimal("1.0") - reliability)))

        # Z-score for confidence level
        # For a 95% CI, (1 + 0.95) / 2 = 0.975, which gives z approx 1.96
        z = to_decimal(scipy.stats.norm.ppf((1 + confidence_level) / 2))

        # Margin of error
        margin = z * sem

        ci_lower = clamp(score - margin)
        ci_upper = clamp(score + margin)

        return ConfidenceInterval(
            point_estimate=score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            sem=sem,
            reliability=reliability,
            evidence_count=n,
            confidence_level=confidence_level,
        )

confidence_calculator = ConfidenceCalculator()

# --- Example Usage: Calculate CI for a sample Org-AI-R score ---
sample_score = to_decimal("75.5")
sample_evidence_count = 15
sample_score_type = "org_air"

sample_ci = confidence_calculator.calculate(
    score=sample_score,
    score_type=sample_score_type,
    evidence_count=sample_evidence_count
)

logger.info(
    "sample_confidence_interval_calculated",
    score=float(sample_score),
    score_type=sample_score_type,
    evidence_count=sample_evidence_count,
    ci_lower=float(sample_ci.ci_lower),
    ci_upper=float(sample_ci.ci_upper),
    sem=float(sample_ci.sem),
    reliability=float(sample_ci.reliability)
)

print(f"Sample Org-AI-R Score: {sample_ci.point_estimate:.2f}")
print(f"Confidence Interval ({sample_ci.confidence_level*100:.0f}%): "
      f"[{sample_ci.ci_lower:.2f}, {sample_ci.ci_upper:.2f}]")
print(f"SEM: {sample_ci.sem:.2f}")
print(f"Reliability (ρ): {sample_ci.reliability:.2f}")
print(f"CI Width: {sample_ci.ci_width:.2f}")
# The OrgAIRCalculator will be defined in a later section, but we can define its internal method here
# for logical flow. This method will be part of the OrgAIRCalculator class.

# Define a temporary placeholder class to demonstrate _calculate_hr before the full OrgAIRCalculator
class OrgAIRCalculatorTemp:
    def __init__(self):
        self.delta = to_decimal(settings.DELTA_POSITION)

    def _calculate_hr(self, baseline: float, position_factor: float) -> HRResult:
        """
        Calculate H^R (Systematic Opportunity) with corrected δ.
        H^R = H^R_base × (1 + δ × PositionFactor)
        Where δ = 0.15 (CORRECTED from 0.5)
        """
        baseline_dec = to_decimal(baseline)
        pf_dec = to_decimal(position_factor)

        hr_score = baseline_dec * (Decimal("1.0") + self.delta * pf_dec)
        hr_score = clamp(hr_score) # Ensure hr_score is within 0-100 range

        logger.info(
            "hr_calculated",
            baseline=float(baseline_dec),
            position_factor=float(pf_dec),
            delta_used=float(self.delta),
            hr_score=float(hr_score)
        )
        return HRResult(
            hr_score=hr_score,
            baseline=baseline_dec,
            position_factor=pf_dec,
            delta_used=self.delta,
        )

# --- Example Usage: Calculate H^R ---
temp_calculator = OrgAIRCalculatorTemp()
sample_hr_baseline = 60.0
sample_position_factor = 0.8 # e.g., slightly conservative position

hr_result = temp_calculator._calculate_hr(sample_hr_baseline, sample_position_factor)

print(f"H^R Baseline: {hr_result.baseline:.2f}")
print(f"Position Factor: {hr_result.position_factor:.2f}")
print(f"Delta Used (δ): {hr_result.delta_used:.2f}")
print(f"Calculated H^R Score: {hr_result.hr_score:.2f}")
# This method will also be part of the OrgAIRCalculator class.

# Extend the temporary placeholder class
class OrgAIRCalculatorTemp2(OrgAIRCalculatorTemp): # Inherit to keep _calculate_hr
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
        TimingFactor clamped to [0.8, 1.2]
        """
        alignment_dec = to_decimal(alignment)

        # Clamp timing factor to [0.8, 1.2]
        timing_dec = clamp(to_decimal(timing), Decimal("0.8"), Decimal("1.2"))

        # Interaction term (V^R * H^R / 100)
        interaction = (vr_score * hr_score) / Decimal("100.0")

        synergy = interaction * alignment_dec * timing_dec
        synergy = clamp(synergy) # Ensure synergy score is within 0-100 range

        logger.info(
            "synergy_calculated",
            vr_score=float(vr_score),
            hr_score=float(hr_score),
            alignment_factor=float(alignment_dec),
            timing_factor=float(timing_dec),
            interaction=float(interaction),
            synergy_score=float(synergy)
        )
        return SynergyResult(
            synergy_score=synergy,
            alignment_factor=alignment_dec,
            timing_factor=timing_dec,
            interaction=interaction,
        )

# --- Example Usage: Calculate Synergy ---
temp_calculator_2 = OrgAIRCalculatorTemp2()
sample_vr_score = to_decimal("70.0")
sample_hr_score_synergy = to_decimal("65.0")
sample_alignment_factor = 0.9
sample_timing_factor_good = 1.1 # Within range
sample_timing_factor_bad = 1.5 # Will be clamped to 1.2

synergy_result_good_timing = temp_calculator_2._calculate_synergy(
    sample_vr_score,
    sample_hr_score_synergy,
    sample_alignment_factor,
    sample_timing_factor_good
)

synergy_result_bad_timing = temp_calculator_2._calculate_synergy(
    sample_vr_score,
    sample_hr_score_synergy,
    sample_alignment_factor,
    sample_timing_factor_bad
)

print(f"Synergy (Good Timing {sample_timing_factor_good:.1f}): {synergy_result_good_timing.synergy_score:.2f} "
      f"(Actual timing used: {synergy_result_good_timing.timing_factor:.2f})")
print(f"Synergy (Bad Timing {sample_timing_factor_bad:.1f}): {synergy_result_bad_timing.synergy_score:.2f} "
      f"(Actual timing used: {synergy_result_bad_timing.timing_factor:.2f})")
class OrgAIRCalculator:
    """Full Org-AI-R calculator with all components."""

    def __init__(self):
        self.alpha = to_decimal(settings.ALPHA_VR_WEIGHT)
        self.beta = to_decimal(settings.BETA_SYNERGY_WEIGHT)
        self.delta = to_decimal(settings.DELTA_POSITION) # For _calculate_hr

    def _calculate_hr(self, baseline: float, position_factor: float) -> HRResult:
        """
        Calculate H^R (Systematic Opportunity) with corrected δ.
        H^R = H^R_base × (1 + δ × PositionFactor)
        Where δ = 0.15 (CORRECTED from 0.5)
        """
        baseline_dec = to_decimal(baseline)
        pf_dec = to_decimal(position_factor)

        hr_score = baseline_dec * (Decimal("1.0") + self.delta * pf_dec)
        hr_score = clamp(hr_score)

        logger.info(
            "hr_calculated_component",
            baseline=float(baseline_dec),
            position_factor=float(pf_dec),
            delta_used=float(self.delta),
            hr_score=float(hr_score)
        )
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
        TimingFactor clamped to [0.8, 1.2]
        """
        alignment_dec = to_decimal(alignment)
        timing_dec = clamp(to_decimal(timing), Decimal("0.8"), Decimal("1.2"))

        interaction = (vr_score * hr_score) / Decimal("100.0")

        synergy = interaction * alignment_dec * timing_dec
        synergy = clamp(synergy)

        logger.info(
            "synergy_calculated_component",
            vr_score=float(vr_score),
            hr_score=float(hr_score),
            alignment_factor=float(alignment_dec),
            timing_factor=float(timing_dec),
            interaction=float(interaction),
            synergy_score=float(synergy)
        )
        return SynergyResult(
            synergy_score=synergy,
            alignment_factor=alignment_dec,
            timing_factor=timing_dec,
            interaction=interaction,
        )

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
        confidence_level: float = 0.95,
    ) -> OrgAIRResult:
        """Calculate complete Org-AI-R score."""
        score_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        logger.info("org_air_calculation_started", company_id=company_id, score_id=score_id)

        # Step 1: Calculate V^R (Idiosyncratic Readiness)
        vr_result = vr_calculator.calculate(dimension_scores, talent_concentration)
        logger.info("vr_calculated_step", vr_score=float(vr_result.vr_score), score_id=score_id)

        # Step 2: Calculate H^R (Systematic Opportunity) with corrected δ
        hr_result = self._calculate_hr(hr_baseline, position_factor)
        logger.info("hr_calculated_step", hr_score=float(hr_result.hr_score), score_id=score_id)

        # Step 3: Calculate Synergy (Interaction of V^R and H^R with Alignment and TimingFactor)
        synergy_result = self._calculate_synergy(
            vr_result.vr_score,
            hr_result.hr_score,
            alignment_factor,
            timing_factor,
        )
        logger.info("synergy_calculated_step", synergy_score=float(synergy_result.synergy_score), score_id=score_id)

        # Step 4: Aggregate Full Org-AI-R Score
        # Org-AI-R = (1-β) × [α×V^R + (1-α)×H^R] + ẞ × Synergy
        one_minus_beta = Decimal("1.0") - self.beta
        weighted_components = (
            self.alpha * vr_result.vr_score +
            (Decimal("1.0") - self.alpha) * hr_result.hr_score
        )

        final_score = (one_minus_beta * weighted_components) + (self.beta * synergy_result.synergy_score)
        final_score = clamp(final_score) # Ensure final score is within 0-100 range

        logger.info("org_air_aggregated_step", final_score=float(final_score), score_id=score_id)

        # Step 5: Calculate SEM-based confidence interval for the final Org-AI-R score
        ci = confidence_calculator.calculate(
            score=final_score,
            score_type="org_air",
            evidence_count=evidence_count,
            confidence_level=confidence_level,
        )
        logger.info(
            "org_air_confidence_interval_calculated_step",
            score=float(final_score),
            ci_lower=float(ci.ci_lower),
            ci_upper=float(ci.ci_upper),
            sem=float(ci.sem),
            reliability=float(ci.reliability),
            score_id=score_id
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

        logger.info("org_air_calculation_completed", **result.to_dict())

        return result

org_air_calculator = OrgAIRCalculator()

# --- Example Usage: Calculate Org-AI-R for a single company ---
sample_company_id = "ORG001"
sample_sector_id = "TECH"
sample_dimension_scores = [70.0, 75.0, 68.0, 80.0]
sample_talent_concentration = 25.0
sample_hr_baseline = 60.0
sample_position_factor = 0.8
sample_alignment_factor = 0.9
sample_timing_factor = 1.1
sample_evidence_count = 15 # More evidence leads to narrower CI

org_air_result = org_air_calculator.calculate(
    company_id=sample_company_id,
    sector_id=sample_sector_id,
    dimension_scores=sample_dimension_scores,
    talent_concentration=sample_talent_concentration,
    hr_baseline=sample_hr_baseline,
    position_factor=sample_position_factor,
    alignment_factor=sample_alignment_factor,
    timing_factor=sample_timing_factor,
    evidence_count=sample_evidence_count
)

print("\n--- Full Org-AI-R Calculation Result (Company 1) ---")
print(f"Company ID: {org_air_result.company_id}")
print(f"Final Org-AI-R Score: {org_air_result.final_score:.2f}")
print(f"  V^R Component: {org_air_result.vr_result.vr_score:.2f}")
print(f"  H^R Component: {org_air_result.hr_result.hr_score:.2f}")
print(f"  Synergy Component: {org_air_result.synergy_result.synergy_score:.2f}")
print(f"Confidence Interval: [{org_air_result.confidence_interval.ci_lower:.2f}, {org_air_result.confidence_interval.ci_upper:.2f}]")
print(f"Confidence Level: {org_air_result.confidence_interval.confidence_level*100:.0f}%")
print(f"SEM: {org_air_result.confidence_interval.sem:.2f}")
print(f"Reliability: {org_air_result.confidence_interval.reliability:.2f}")
print(f"Evidence Count: {org_air_result.confidence_interval.evidence_count}")
print(f"Parameters Version: {org_air_result.parameter_version}")
# --- Define Multiple Synthetic Scenarios ---
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
    },
    {
        "company_id": "TRADITIONAL_CO",
        "sector_id": "MANUFACTURING",
        "dimension_scores": [45.0, 50.0, 40.0, 55.0],
        "talent_concentration": 15.0,
        "hr_baseline": 40.0,
        "position_factor": 0.6,
        "alignment_factor": 0.7,
        "timing_factor": 0.8, # Clamped if lower
        "evidence_count": 5,  # Low evidence
        "confidence_tier": 3,
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
    },
]

all_results = []
for scenario in scenarios:
    result = org_air_calculator.calculate(**scenario)
    all_results.append(result.to_dict())

results_df = pd.DataFrame(all_results)
results_df['org_air_score'] = results_df['final_score'] # for consistent column naming
results_df['ci_error'] = (results_df['ci_upper'] - results_df['ci_lower']) / 2 # Half width for error bar

print("\n--- Aggregated Results Across Scenarios ---")
print(results_df[['company_id', 'org_air_score', 'ci_lower', 'ci_upper', 'sem', 'reliability', 'evidence_count']].round(2))

# --- Visualization 1: Org-AI-R Scores with Confidence Intervals ---
plt.figure(figsize=(12, 7))
sns.barplot(x='company_id', y='org_air_score', data=results_df, palette='viridis')
plt.errorbar(
    x=results_df['company_id'],
    y=results_df['org_air_score'],
    yerr=results_df['ci_error'],
    fmt='none',
    c='black',
    capsize=5,
    label='95% Confidence Interval'
)
plt.xlabel("Company ID")
plt.ylabel("Org-AI-R Score")
plt.title("Org-AI-R Scores with SEM-Based 95% Confidence Intervals Across Companies")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# --- Visualization 2: Reliability vs. Evidence Count (Illustrating Spearman-Brown) ---
# Generate a range of evidence counts
evidence_counts = np.arange(1, 31)
default_item_corr = float(ConfidenceCalculator.DEFAULT_ITEM_CORRELATION)
reliabilities = []

for n in evidence_counts:
    rho = (n * default_item_corr) / (1 + (n - 1) * default_item_corr)
    reliabilities.append(min(rho, 0.99)) # Cap at 0.99

plt.figure(figsize=(10, 6))
plt.plot(evidence_counts, reliabilities, marker='o', linestyle='-', color='skyblue')
plt.axhline(y=0.7, color='red', linestyle='--', label='Acceptable Reliability (0.7)')
plt.axhline(y=0.9, color='green', linestyle='--', label='High Reliability (0.9)')
plt.xlabel("Number of Evidence Items (n)")
plt.ylabel("Estimated Reliability (ρ)")
plt.title("Impact of Evidence Count on Score Reliability (Spearman-Brown Prophecy)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(0, 1)
plt.show()