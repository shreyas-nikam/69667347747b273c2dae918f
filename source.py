"""
org_air_core.py (function-driven refactor)

Drop this into a module and import the functions/classes from app.py.
Core design goals:
- No side effects at import time (no sample runs, no plots, no prints).
- All “demo / scenarios / plotting” moved into callable functions.
- Dependency injection for settings + calculators + logger (app.py can control config).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, List, Dict, Any, Sequence, Tuple
import uuid
import math

import scipy.stats
import structlog

# --------------------------------------------------------------------------------------
# Decimal + Utilities
# --------------------------------------------------------------------------------------


def configure_decimal_precision(prec: int = 10) -> None:
    """Configure Decimal precision for financial-grade accuracy."""
    getcontext().prec = prec


def to_decimal(value) -> Decimal:
    """Converts a float/int/str/Decimal to Decimal, handling None safely."""
    if value is None:
        return Decimal("0.0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def clamp(
    value: Decimal,
    min_val: Decimal = Decimal("0.0"),
    max_val: Decimal = Decimal("100.0"),
) -> Decimal:
    """Clamps a Decimal value within a specified range."""
    return max(min_val, min(max_val, value))


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    alpha_vr_weight: float = 0.60
    beta_synergy_weight: float = 0.12
    delta_position: float = 0.15
    param_version: str = "1.0.0"


def default_settings() -> Settings:
    """Factory for default settings (handy for app.py)."""
    return Settings()


# --------------------------------------------------------------------------------------
# Dataclasses: Results
# --------------------------------------------------------------------------------------

@dataclass
class VRResult:
    vr_score: Decimal
    dimension_scores_raw: List[Decimal]
    talent_concentration_raw: Decimal


@dataclass
class ConfidenceInterval:
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
    hr_score: Decimal
    baseline: Decimal
    position_factor: Decimal
    delta_used: Decimal


@dataclass
class SynergyResult:
    synergy_score: Decimal
    alignment_factor: Decimal
    timing_factor: Decimal
    interaction: Decimal


@dataclass
class OrgAIRResult:
    score_id: str
    company_id: str
    sector_id: str
    timestamp: datetime
    final_score: Decimal
    vr_result: VRResult
    hr_result: HRResult
    synergy_result: SynergyResult
    confidence_interval: ConfidenceInterval
    confidence_tier: int
    alpha: Decimal
    beta: Decimal
    parameter_version: str

    def to_dict(self) -> Dict[str, Any]:
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


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def configure_structlog(dev_console: bool = True) -> structlog.stdlib.BoundLogger:
    """
    Configure structlog and return a logger.
    app.py can call this once and pass logger into functions below.

    dev_console=True uses ConsoleRenderer. Switch to JSONRenderer in production.
    """
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer() if dev_console else structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


# --------------------------------------------------------------------------------------
# Calculators
# --------------------------------------------------------------------------------------

class VRCalculator:
    """
    Placeholder V^R calculator: average of dimension scores * talent factor.
    Talent factor = 1 + (talent_concentration / 100).
    """

    def calculate(self, dimension_scores: Sequence[float], talent_concentration: float) -> VRResult:
        dim_scores_dec = [to_decimal(s) for s in (dimension_scores or [])]
        talent_dec = to_decimal(talent_concentration)

        if not dim_scores_dec:
            avg_dim_score = Decimal("0.0")
        else:
            avg_dim_score = sum(dim_scores_dec) / \
                to_decimal(len(dim_scores_dec))

        talent_factor = Decimal("1.0") + (talent_dec / Decimal("100.0"))
        vr_score = clamp(avg_dim_score * talent_factor)

        return VRResult(
            vr_score=vr_score,
            dimension_scores_raw=dim_scores_dec,
            talent_concentration_raw=talent_dec,
        )


class ConfidenceCalculator:
    """Calculates SEM-based confidence intervals."""

    POPULATION_SD: Dict[str, Decimal] = {
        "vr": to_decimal("15.0"),
        "hr": to_decimal("12.0"),
        "synergy": to_decimal("10.0"),
        "org_air": to_decimal("14.0"),
    }
    DEFAULT_ITEM_CORRELATION: Decimal = to_decimal("0.30")

    def calculate(
        self,
        score: Decimal,
        score_type: str,
        evidence_count: int,
        item_correlation: Optional[float] = None,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        item_corr_dec = (
            to_decimal(
                item_correlation) if item_correlation is not None else self.DEFAULT_ITEM_CORRELATION
        )

        n = max(int(evidence_count), 1)

        # Spearman-Brown prophecy formula:
        # ρ = (n * r) / (1 + (n-1) * r)
        reliability = (to_decimal(n) * item_corr_dec) / (
            Decimal("1.0") + (to_decimal(n) - Decimal("1.0")) * item_corr_dec
        )
        reliability = clamp(reliability, Decimal("0.0"), Decimal("0.99"))

        sigma = self.POPULATION_SD.get(score_type, to_decimal("15.0"))

        sem = sigma * \
            to_decimal(math.sqrt(float(Decimal("1.0") - reliability)))

        z = to_decimal(scipy.stats.norm.ppf((1 + confidence_level) / 2))
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


class OrgAIRCalculator:
    """Full Org-AI-R calculator with all components."""

    def __init__(
        self,
        settings: Settings,
        vr_calculator: VRCalculator,
        confidence_calculator: ConfidenceCalculator,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        self.settings = settings
        self.vr_calculator = vr_calculator
        self.confidence_calculator = confidence_calculator
        self.logger = logger

        self.alpha = to_decimal(settings.alpha_vr_weight)
        self.beta = to_decimal(settings.beta_synergy_weight)
        self.delta = to_decimal(settings.delta_position)

    def _log(self, event: str, **kwargs) -> None:
        if self.logger is not None:
            self.logger.info(event, **kwargs)

    def calculate_hr(self, baseline: float, position_factor: float) -> HRResult:
        baseline_dec = to_decimal(baseline)
        pf_dec = to_decimal(position_factor)

        hr_score = clamp(baseline_dec * (Decimal("1.0") + self.delta * pf_dec))

        self._log(
            "hr_calculated",
            baseline=float(baseline_dec),
            position_factor=float(pf_dec),
            delta_used=float(self.delta),
            hr_score=float(hr_score),
        )

        return HRResult(
            hr_score=hr_score,
            baseline=baseline_dec,
            position_factor=pf_dec,
            delta_used=self.delta,
        )

    def calculate_synergy(
        self,
        vr_score: Decimal,
        hr_score: Decimal,
        alignment: float,
        timing: float,
    ) -> SynergyResult:
        alignment_dec = to_decimal(alignment)
        timing_dec = clamp(to_decimal(timing), Decimal("0.8"), Decimal("1.2"))

        interaction = (vr_score * hr_score) / Decimal("100.0")
        synergy_score = clamp(interaction * alignment_dec * timing_dec)

        self._log(
            "synergy_calculated",
            vr_score=float(vr_score),
            hr_score=float(hr_score),
            alignment_factor=float(alignment_dec),
            timing_factor=float(timing_dec),
            interaction=float(interaction),
            synergy_score=float(synergy_score),
        )

        return SynergyResult(
            synergy_score=synergy_score,
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
        score_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        self._log("org_air_calculation_started", company_id=company_id,
                  sector_id=sector_id, score_id=score_id)

        vr_result = self.vr_calculator.calculate(
            dimension_scores, talent_concentration)
        self._log("vr_calculated", score_id=score_id,
                  vr_score=float(vr_result.vr_score))

        hr_result = self.calculate_hr(hr_baseline, position_factor)
        self._log("hr_calculated_step", score_id=score_id,
                  hr_score=float(hr_result.hr_score))

        synergy_result = self.calculate_synergy(
            vr_score=vr_result.vr_score,
            hr_score=hr_result.hr_score,
            alignment=alignment_factor,
            timing=timing_factor,
        )
        self._log("synergy_calculated_step", score_id=score_id,
                  synergy_score=float(synergy_result.synergy_score))

        # Org-AI-R = (1-β) × [α×V^R + (1-α)×H^R] + β × Synergy
        one_minus_beta = Decimal("1.0") - self.beta
        weighted_components = (self.alpha * vr_result.vr_score) + \
            ((Decimal("1.0") - self.alpha) * hr_result.hr_score)
        final_score = clamp((one_minus_beta * weighted_components) +
                            (self.beta * synergy_result.synergy_score))

        self._log("org_air_aggregated", score_id=score_id,
                  final_score=float(final_score))

        ci = self.confidence_calculator.calculate(
            score=final_score,
            score_type="org_air",
            evidence_count=evidence_count,
            confidence_level=confidence_level,
        )

        self._log(
            "org_air_confidence_interval_calculated",
            score_id=score_id,
            score=float(final_score),
            ci_lower=float(ci.ci_lower),
            ci_upper=float(ci.ci_upper),
            sem=float(ci.sem),
            reliability=float(ci.reliability),
            evidence_count=ci.evidence_count,
            confidence_level=ci.confidence_level,
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
            parameter_version=self.settings.param_version,
        )

        self._log("org_air_calculation_completed", **result.to_dict())
        return result


# --------------------------------------------------------------------------------------
# Factory / Wiring helpers (app.py-friendly)
# --------------------------------------------------------------------------------------

@dataclass
class CalculatorBundle:
    settings: Settings
    logger: structlog.stdlib.BoundLogger
    vr_calculator: VRCalculator
    confidence_calculator: ConfidenceCalculator
    org_air_calculator: OrgAIRCalculator


def build_calculators(
    settings: Optional[Settings] = None,
    logger: Optional[structlog.stdlib.BoundLogger] = None,
    decimal_precision: int = 10,
    dev_console_logger: bool = True,
) -> CalculatorBundle:
    """
    One-stop wiring for app.py.
    Returns a bundle containing settings + calculators + logger.
    """
    configure_decimal_precision(decimal_precision)
    s = settings or default_settings()
    lg = logger or configure_structlog(dev_console=dev_console_logger)

    vr = VRCalculator()
    cc = ConfidenceCalculator()
    org = OrgAIRCalculator(settings=s, vr_calculator=vr,
                           confidence_calculator=cc, logger=lg)

    return CalculatorBundle(
        settings=s,
        logger=lg,
        vr_calculator=vr,
        confidence_calculator=cc,
        org_air_calculator=org,
    )


# --------------------------------------------------------------------------------------
# Scenario + Reporting helpers (no side effects)
# --------------------------------------------------------------------------------------

def run_scenarios(
    org_air_calculator: OrgAIRCalculator,
    scenarios: Sequence[Dict[str, Any]],
) -> List[OrgAIRResult]:
    """Run multiple scenarios and return the list of OrgAIRResult objects."""
    results: List[OrgAIRResult] = []
    for scenario in scenarios:
        results.append(org_air_calculator.calculate(**scenario))
    return results


def results_to_records(results: Sequence[OrgAIRResult]) -> List[Dict[str, Any]]:
    """Convert results to dict records (easy for pandas / JSON)."""
    return [r.to_dict() for r in results]


def compute_ci_half_width(record: Dict[str, Any]) -> float:
    """Convenience: half-width error for error bars."""
    return (record["ci_upper"] - record["ci_lower"]) / 2.0


# --------------------------------------------------------------------------------------
# Optional plotting helpers (matplotlib only; call from app.py if needed)
# --------------------------------------------------------------------------------------

def plot_org_air_scores_with_ci_matplotlib(records: Sequence[Dict[str, Any]]) -> None:
    """
    Plot bar chart of Org-AI-R scores with CI error bars.
    NOTE: This function imports matplotlib lazily so the core module stays lightweight.
    """
    import matplotlib.pyplot as plt  # lazy import

    company_ids = [r["company_id"] for r in records]
    scores = [r["final_score"] for r in records]
    yerr = [compute_ci_half_width(r) for r in records]

    plt.figure(figsize=(12, 7))
    plt.bar(company_ids, scores)
    plt.errorbar(company_ids, scores, yerr=yerr, fmt="none", capsize=5)
    plt.xlabel("Company ID")
    plt.ylabel("Org-AI-R Score")
    plt.title("Org-AI-R Scores with SEM-Based Confidence Intervals")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_reliability_vs_evidence_count(
    item_correlation: float = 0.30,
    max_n: int = 30,
) -> None:
    """Illustrate Spearman-Brown effect on reliability vs evidence count."""
    import matplotlib.pyplot as plt  # lazy import

    xs = list(range(1, max_n + 1))
    reliabilities: List[float] = []
    r = float(item_correlation)

    for n in xs:
        rho = (n * r) / (1 + (n - 1) * r)
        reliabilities.append(min(rho, 0.99))

    plt.figure(figsize=(10, 6))
    plt.plot(xs, reliabilities, marker="o", linestyle="-")
    plt.axhline(y=0.7, linestyle="--")
    plt.axhline(y=0.9, linestyle="--")
    plt.xlabel("Number of Evidence Items (n)")
    plt.ylabel("Estimated Reliability (ρ)")
    plt.title("Impact of Evidence Count on Score Reliability (Spearman-Brown)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1)
    plt.show()
