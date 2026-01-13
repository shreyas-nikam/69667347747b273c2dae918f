# QuLab: H^R, Synergy & Full Org-AI-R with SEM-Based CI

## Project Description

This Streamlit application, "QuLab: H^R, Synergy & Full Org-AI-R with SEM-Based CI", serves as a lab project focusing on the implementation and validation of an Organizational AI-Readiness (Org-AI-R) score. Designed as a software developer's workflow, it demonstrates how to build a robust, financial-grade AI-Readiness assessment system. The application emphasizes precise aggregation of various components (`V^R`, `H^R`, Synergy), the crucial role of Standard Error of Measurement (SEM)-based Confidence Intervals (CIs) for reliability, and the importance of audit trails.

The project aims to provide a practical understanding of:
*   Implementing complex scoring logic with high accuracy.
*   Applying statistical concepts like SEM and Spearman-Brown prophecy to quantify measurement uncertainty.
*   Visualizing results to communicate reliability and facilitate strategic decision-making.

This tool empowers organizations to assess their AI readiness with a reliable and auditable metric, moving beyond subjective evaluations to data-driven insights.

## Key Objectives of the Lab Project

As explored within the application, the key objectives for a software developer building this system include:
*   **Understanding SEM**: Grasping the Standard Error of Measurement formula and its components.
*   **Addressing CI Issues**: Explaining why fixed-width confidence intervals are problematic.
*   **Implementing SEM-Based CIs**: Applying proper SEM-based confidence intervals.
*   **Analyzing Reliability**: Comparing reliability across varying evidence counts.
*   **Evaluating CI Accuracy**: Assessing the calibration accuracy of confidence intervals.
*   **Designing Audit-Ready Pipelines**: Creating a scoring pipeline capable of comprehensive audit trails.

## Features

This application provides the following core functionalities:

*   **Org-AI-R Score Calculation**:
    *   Calculates `V^R` (Idiosyncratic Readiness) based on dimension scores and talent concentration.
    *   Calculates `H^R` (Systematic Opportunity) using a baseline score and a position adjustment factor.
    *   Calculates `Synergy` as an interaction effect between `V^R`, `H^R`, alignment, and timing factors.
    *   Aggregates these components into a final `Org-AI-R` score using a weighted formula.
*   **SEM-Based Confidence Interval (CI) Calculation**:
    *   Implements the Standard Error of Measurement (SEM) using population standard deviation and score reliability.
    *   Estimates reliability using the **Spearman-Brown prophecy formula**, which considers the number of evidence items and average inter-item correlation.
    *   Generates dynamic confidence intervals that reflect the precision of the score based on evidence.
*   **Interactive Input Parameters**: A user-friendly Streamlit interface allows for entering custom parameters for company details, `V^R`, `H^R`, Synergy, and Confidence Interval factors.
*   **Detailed Results & Auditability**: Displays the final Org-AI-R score, its components, and a full JSON output of the calculation result for audit purposes.
*   **Scenario Analysis**: Pre-configured scenarios for multiple hypothetical companies to demonstrate the system's behavior under different conditions.
*   **Data Visualization**:
    *   **Org-AI-R Scores with CIs**: Bar chart visualizing Org-AI-R scores with SEM-based error bars, highlighting how confidence intervals vary with evidence count.
    *   **Reliability vs. Evidence Count**: Plot illustrating the Spearman-Brown prophecy, showing how measurement reliability improves with more evidence.
*   **Precision Arithmetic**: Utilizes Python's `Decimal` module for financial-grade accuracy in calculations (as hinted in project objectives).
*   **Structured Logging**: Designed to integrate `structlog` for comprehensive, audit-ready logging of calculation steps and results (as hinted in project objectives).

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the files):**

    ```bash
    git clone https://github.com/your-username/quair-streamlit-lab.git
    cd quair-streamlit-lab
    ```
    *(Replace `your-username/quair-streamlit-lab.git` with the actual repository URL if available)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the project root with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scipy
    structlog # For structured logging (as mentioned in app intro)
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` is present:**
    The application relies on a `source.py` file containing the core logic (`org_air_calculator`, `confidence_calculator`, etc.). Make sure this file is in the same directory as `app.py`. A placeholder might look like this (you'll need to fill in the actual logic based on the lab project's requirements):

    ```python
    # source.py (Example structure - actual implementation will be provided or developed)

    from dataclasses import dataclass
    from decimal import Decimal, getcontext
    import math
    import structlog
    from scipy.stats import norm

    # Configure Decimal precision for financial-grade calculations
    getcontext().prec = 10

    # Initialize structured logger
    log = structlog.get_logger()

    # Constants
    ALPHA_VR_WEIGHT = Decimal('0.60') # Weight for VR in main Org-AIR formula
    BETA_SYNERGY_WEIGHT = Decimal('0.12') # Weight for Synergy in main Org-AIR formula
    DELTA_POSITION_ADJUSTMENT = Decimal('0.15') # Position adjustment factor for H^R
    TIMING_FACTOR_MIN = Decimal('0.8')
    TIMING_FACTOR_MAX = Decimal('1.2')

    # Default values for Confidence Calculator
    POPULATION_STD_DEV = Decimal('10.0') # Example standard deviation for scores
    DEFAULT_ITEM_CORRELATION = Decimal('0.3') # Average inter-item correlation

    @dataclass
    class ConfidenceIntervalResult:
        point_estimate: Decimal
        ci_lower: Decimal
        ci_upper: Decimal
        sem: Decimal
        reliability: Decimal
        evidence_count: int
        confidence_level: Decimal
        description: str = ""

    @dataclass
    class VRResult:
        average_dimension_score: Decimal
        talent_concentration: Decimal
        vr_score: Decimal
        description: str = ""

    @dataclass
    class HRResult:
        hr_baseline: Decimal
        position_factor: Decimal
        hr_score: Decimal
        description: str = ""

    @dataclass
    class SynergyResult:
        vr_score: Decimal
        hr_score: Decimal
        alignment_factor: Decimal
        timing_factor: Decimal
        synergy_score: Decimal
        description: str = ""

    @dataclass
    class OrgAIRResult:
        company_id: str
        sector_id: str
        final_score: Decimal
        confidence_interval: ConfidenceIntervalResult
        vr_result: VRResult
        hr_result: HRResult
        synergy_result: SynergyResult
        confidence_tier: int
        parameter_version: str = "1.0.0" # Example versioning for parameters

        def to_dict(self):
            return {
                "company_id": self.company_id,
                "sector_id": self.sector_id,
                "final_score": float(self.final_score),
                "confidence_interval": {
                    "point_estimate": float(self.confidence_interval.point_estimate),
                    "ci_lower": float(self.confidence_interval.ci_lower),
                    "ci_upper": float(self.confidence_interval.ci_upper),
                    "sem": float(self.confidence_interval.sem),
                    "reliability": float(self.confidence_interval.reliability),
                    "evidence_count": self.confidence_interval.evidence_count,
                    "confidence_level": float(self.confidence_interval.confidence_level),
                    "description": self.confidence_interval.description
                },
                "vr_result": {
                    "average_dimension_score": float(self.vr_result.average_dimension_score),
                    "talent_concentration": float(self.vr_result.talent_concentration),
                    "vr_score": float(self.vr_result.vr_score),
                    "description": self.vr_result.description
                },
                "hr_result": {
                    "hr_baseline": float(self.hr_result.hr_baseline),
                    "position_factor": float(self.hr_result.position_factor),
                    "hr_score": float(self.hr_result.hr_score),
                    "description": self.hr_result.description
                },
                "synergy_result": {
                    "vr_score": float(self.synergy_result.vr_score),
                    "hr_score": float(self.synergy_result.hr_score),
                    "alignment_factor": float(self.synergy_result.alignment_factor),
                    "timing_factor": float(self.synergy_result.timing_factor),
                    "synergy_score": float(self.synergy_result.synergy_score),
                    "description": self.synergy_result.description
                },
                "confidence_tier": self.confidence_tier,
                "parameter_version": self.parameter_version
            }


    class ConfidenceCalculator:
        DEFAULT_POPULATION_STD_DEV = POPULATION_STD_DEV
        DEFAULT_ITEM_CORRELATION = DEFAULT_ITEM_CORRELATION

        def calculate_reliability(self, evidence_count: int, average_item_correlation: Decimal = DEFAULT_ITEM_CORRELATION) -> Decimal:
            n = Decimal(evidence_count)
            r = average_item_correlation
            if n <= 0:
                return Decimal('0.0') # Or raise an error
            if n == 1: # Reliability of a single item is its inter-item correlation
                return r

            # Spearman-Brown prophecy formula
            rho = (n * r) / (Decimal('1') + (n - Decimal('1')) * r)
            return min(rho, Decimal('0.999')) # Cap reliability at a very high value

        def calculate_sem(self, population_std_dev: Decimal, reliability: Decimal) -> Decimal:
            if reliability >= Decimal('1.0'):
                return Decimal('0.0')
            if reliability < Decimal('0.0'):
                reliability = Decimal('0.0') # Clamp reliability to non-negative
            sem = population_std_dev * (Decimal('1') - reliability).sqrt()
            return sem

        def calculate_confidence_interval(self,
                                        score: Decimal,
                                        evidence_count: int,
                                        confidence_level: Decimal = Decimal('0.95'),
                                        population_std_dev: Decimal = DEFAULT_POPULATION_STD_DEV,
                                        average_item_correlation: Decimal = DEFAULT_ITEM_CORRELATION) -> ConfidenceIntervalResult:
            log.info("Calculating confidence interval", score=score, evidence_count=evidence_count, confidence_level=confidence_level)

            reliability = self.calculate_reliability(evidence_count, average_item_correlation)
            sem = self.calculate_sem(population_std_dev, reliability)

            # Z-score for given confidence level
            # norm.ppf uses float, convert back and forth
            z_score = Decimal(norm.ppf(float(Decimal('1') - (Decimal('1') - confidence_level) / Decimal('2'))))

            margin_of_error = z_score * sem
            ci_lower = score - margin_of_error
            ci_upper = score + margin_of_error

            # Clamp CI to 0-100 range
            ci_lower = max(Decimal('0.0'), ci_lower)
            ci_upper = min(Decimal('100.0'), ci_upper)


            log.info("Confidence interval calculated", ci_lower=ci_lower, ci_upper=ci_upper, sem=sem, reliability=reliability)
            return ConfidenceIntervalResult(
                point_estimate=score,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sem=sem,
                reliability=reliability,
                evidence_count=evidence_count,
                confidence_level=confidence_level,
                description=f"{float(confidence_level*100):.0f}% CI based on SEM and {evidence_count} evidence items."
            )

    class OrgAIRCalculator:
        def __init__(self):
            self.confidence_calculator = ConfidenceCalculator()

        def calculate_vr(self, dimension_scores: list[float], talent_concentration: float) -> VRResult:
            log.info("Calculating V^R", dimension_scores=dimension_scores, talent_concentration=talent_concentration)
            avg_dim_score = Decimal(sum(dimension_scores) / len(dimension_scores)) if dimension_scores else Decimal('0.0')
            talent_factor = Decimal(talent_concentration) / Decimal('100.0')
            vr_score = (avg_dim_score * Decimal('0.7')) + (Decimal('100.0') * talent_factor * Decimal('0.3')) # Example weighting
            vr_score = min(Decimal('100.0'), max(Decimal('0.0'), vr_score))
            log.info("V^R calculated", vr_score=vr_score)
            return VRResult(avg_dim_score, Decimal(talent_concentration), vr_score, "Idiosyncratic Readiness Score")

        def calculate_hr(self, hr_baseline: float, position_factor: float) -> HRResult:
            log.info("Calculating H^R", hr_baseline=hr_baseline, position_factor=position_factor)
            hr_base_dec = Decimal(hr_baseline)
            pos_factor_dec = Decimal(position_factor)
            hr_score = hr_base_dec * (Decimal('1') + DELTA_POSITION_ADJUSTMENT * pos_factor_dec)
            hr_score = min(Decimal('100.0'), max(Decimal('0.0'), hr_score))
            log.info("H^R calculated", hr_score=hr_score)
            return HRResult(hr_base_dec, pos_factor_dec, hr_score, "Systematic Opportunity Score")

        def calculate_synergy(self, vr_score: Decimal, hr_score: Decimal, alignment_factor: float, timing_factor: float) -> SynergyResult:
            log.info("Calculating Synergy", vr_score=vr_score, hr_score=hr_score, alignment_factor=alignment_factor, timing_factor=timing_factor)
            align_factor_dec = Decimal(alignment_factor)
            timing_factor_dec = Decimal(timing_factor)
            # Clamp timing factor
            clamped_timing_factor = max(TIMING_FACTOR_MIN, min(TIMING_FACTOR_MAX, timing_factor_dec))

            synergy_score = (vr_score * hr_score / Decimal('100.0')) * align_factor_dec * clamped_timing_factor
            synergy_score = min(Decimal('100.0'), max(Decimal('0.0'), synergy_score)) # Synergy can also be capped
            log.info("Synergy calculated", synergy_score=synergy_score)
            return SynergyResult(vr_score, hr_score, align_factor_dec, clamped_timing_factor, synergy_score, "Synergy Score")

        def calculate(self,
                    company_id: str,
                    sector_id: str,
                    dimension_scores: list[float],
                    talent_concentration: float,
                    hr_baseline: float,
                    position_factor: float,
                    alignment_factor: float,
                    timing_factor: float,
                    evidence_count: int,
                    confidence_tier: int,
                    confidence_level: float) -> OrgAIRResult:

            log.info("Starting Org-AI-R calculation", company_id=company_id, sector_id=sector_id)

            # Convert all float inputs to Decimal for precision
            dec_talent_concentration = Decimal(str(talent_concentration))
            dec_hr_baseline = Decimal(str(hr_baseline))
            dec_position_factor = Decimal(str(position_factor))
            dec_alignment_factor = Decimal(str(alignment_factor))
            dec_timing_factor = Decimal(str(timing_factor))
            dec_confidence_level = Decimal(str(confidence_level))

            # 1. Calculate V^R
            vr_result = self.calculate_vr(dimension_scores, float(dec_talent_concentration))

            # 2. Calculate H^R
            hr_result = self.calculate_hr(float(dec_hr_baseline), float(dec_position_factor))

            # 3. Calculate Synergy
            synergy_result = self.calculate_synergy(vr_result.vr_score, hr_result.hr_score, float(dec_alignment_factor), float(dec_timing_factor))

            # 4. Aggregate Org-AI-R
            org_air_score = (Decimal('1') - BETA_SYNERGY_WEIGHT) * (ALPHA_VR_WEIGHT * vr_result.vr_score + (Decimal('1') - ALPHA_VR_WEIGHT) * hr_result.hr_score) + BETA_SYNERGY_WEIGHT * synergy_result.synergy_score
            org_air_score = min(Decimal('100.0'), max(Decimal('0.0'), org_air_score)) # Ensure score is within 0-100

            # 5. Calculate Confidence Interval
            ci_result = self.confidence_calculator.calculate_confidence_interval(
                score=org_air_score,
                evidence_count=evidence_count,
                confidence_level=dec_confidence_level
            )

            log.info("Org-AI-R calculation complete", final_score=org_air_score)

            return OrgAIRResult(
                company_id=company_id,
                sector_id=sector_id,
                final_score=org_air_score,
                confidence_interval=ci_result,
                vr_result=vr_result,
                hr_result=hr_result,
                synergy_result=synergy_result,
                confidence_tier=confidence_tier
            )

    org_air_calculator = OrgAIRCalculator()
    confidence_calculator = ConfidenceCalculator()
    ```

## Usage

1.  **Run the Streamlit application:**
    Open your terminal or command prompt, navigate to the project directory, and execute:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Navigate the Application:**
    *   Use the **"Org-AI-R System Navigation"** sidebar to switch between pages.
    *   **Introduction**: Provides an overview of the project, its objectives, and key concepts.
    *   **Org-AI-R Calculator**:
        *   Input various parameters (Company ID, Dimension Scores, Talent Concentration, HR Baseline, etc.) into the form.
        *   Click **"Calculate Org-AI-R Score"** to compute the score and its confidence interval.
        *   Review the detailed results, including individual components, confidence interval specifics, and the full audit-ready JSON output.
    *   **Scenario Analysis & Visualization**:
        *   Click **"Run Scenario Analysis for Multiple Companies"** to process a set of predefined company scenarios.
        *   View a table of aggregated results.
        *   Examine the interactive plots: "Org-AI-R Scores with Confidence Intervals" and "Reliability vs. Evidence Count" to understand how scores and their precision vary across different inputs and evidence levels.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains core calculation logic (OrgAIRCalculator, ConfidenceCalculator)
├── requirements.txt        # Python dependencies
└── README.md               # This README file
```

## Technology Stack

*   **Frontend / UI**: [Streamlit](https://streamlit.io/)
*   **Core Logic**: Python 3.8+
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Statistical Analysis**: [SciPy (specifically `scipy.stats.norm`)](https://scipy.org/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Precision Arithmetic**: Python's built-in `Decimal` module for financial-grade accuracy.
*   **Structured Logging**: [structlog](https://www.structlog.org/) (integrated for auditability).

## Contributing

This is a lab project, primarily for learning and demonstration. However, if you find any bugs or have suggestions for improvements, feel free to:

1.  Open an issue on the GitHub repository.
2.  Fork the repository and submit a pull request with your changes.

When contributing, please ensure your code adheres to good practices, includes docstrings, and passes any existing tests (or new tests if applicable).

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.

```
MIT License

Copyright (c) [Year] [Your Name/QuantUniversity]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For questions, feedback, or collaborations, please reach out to:

*   **GitHub**: [Your GitHub Profile](https://github.com/your-username)
*   **QuantUniversity**: [www.quantuniversity.com](https://www.quantuniversity.com/) (for project context)

```
