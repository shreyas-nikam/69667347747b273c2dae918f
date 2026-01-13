
from streamlit.testing.v1 import AppTest
import pytest

# Assuming 'app.py' is the filename of the Streamlit application code provided.
APP_FILE = "app.py"


def test_initial_page_and_title():
    """
    Test that the app loads correctly and displays the Introduction page
    with the correct title and initial content.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Check page config title
    assert at.main[0].title.value == "QuLab: H^R, Synergy & Full Org-AIR with SEM-Based CI"

    # Check initial page content (Introduction)
    # The first markdown block contains the main welcome message and the initial header
    assert at.markdown[0].value.startswith("# Org-AI-R Score Calculation and Validation")
    assert "Welcome to Week 6 of developing the PE-Org AIR System!" in at.markdown[0].value
    # The second markdown block contains the "Key Objectives" header and its list items
    assert "## Key Objectives" in at.markdown[1].value


def test_sidebar_navigation_to_calculator():
    """
    Test navigation to the 'Org-AI-R Calculator' page and verify its content.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Select "Org-AI-R Calculator" in the sidebar radio button
    at.sidebar.radio[0].set_value("Org-AI-R Calculator").run()

    # Verify the title and introductory markdowns of the calculator page
    assert at.markdown[0].value == "# Org-AI-R Calculator: Company Assessment"
    assert "Input Parameters for Org-AI-R Calculation" in at.markdown[1].value
    assert "Core Formulas" in at.markdown[2].value
    # Ensure form subheaders are present within the form
    assert at.form[0].subheader[0].value == "Company Details"
    assert at.form[0].subheader[4].value == "Confidence Interval Factors"
    # Ensure all default input widgets are present and have their default values
    assert at.text_input[0].value == "ACME_CORP"
    assert at.text_area[0].value == "70.0, 75.0, 68.0, 80.0"
    assert at.number_input[0].value == 25.0  # Talent Concentration


def test_org_air_calculation_success():
    """
    Test the Org-AI-R Calculator's functionality:
    - Inputting parameters
    - Clicking calculate button
    - Verifying success message and result display.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Navigate to the calculator page
    at.sidebar.radio[0].set_value("Org-AI-R Calculator").run()

    # Simulate inputting values (overriding some defaults to confirm interaction)
    at.text_input[0].set_value("TEST_CORP").run()  # Company ID
    at.text_area[0].set_value("80.0, 82.0, 78.0, 85.0").run()  # Dimension Scores
    at.number_input[0].set_value(30.0).run()  # Talent Concentration
    at.number_input[5].set_value(20).run()    # Evidence Count
    at.number_input[7].set_value(0.90).run()  # Confidence Level

    # Click the calculate button in the form
    at.form[0].button[0].click().run()

    # Verify success message
    assert at.success[0].value == "Org-AI-R Score Calculated Successfully!"

    # Verify that results are displayed (checking for key metrics and headers)
    # There are 9 descriptive markdown blocks before the results section
    assert at.markdown[9].value == "### 2. Org-AI-R Calculation Result"
    assert at.metric[0].label == "Final Org-AI-R Score"
    assert at.metric[1].label == "V^R Component"
    assert at.metric[2].label == "H^R Component"
    assert at.metric[3].label == "Synergy Component"

    # Verify confidence interval details are present (these subheaders and markdowns are outside the form)
    assert at.subheader[0].value == "Confidence Interval Details"
    assert at.markdown[10].value.startswith("**Point Estimate**:")
    assert at.markdown[11].value.startswith("**Confidence Interval")
    assert at.markdown[12].value.startswith("**SEM (Standard Error of Measurement)**:")
    assert at.markdown[13].value.startswith("**Reliability (ρ)**:")
    assert at.markdown[14].value.startswith("**Evidence Count (n)**:")
    assert at.markdown[15].value.startswith("**Confidence Level**:")
    assert at.markdown[16].value.startswith("**Confidence Tier**:")
    assert at.markdown[17].value.startswith("**Parameters Version**:")

    # Verify the full result object is displayed as JSON
    assert at.subheader[1].value == "Full Result Object (for Auditability)"
    assert at.json[0].value is not None
    assert "final_score" in at.json[0].value
    assert at.session_state.org_air_result is not None
    assert at.session_state.org_air_result.company_id == "TEST_CORP"


def test_org_air_calculation_error_handling():
    """
    Test error handling for invalid input in the Org-AI-R Calculator.
    Specifically, a malformed dimension_scores string.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Navigate to the calculator page
    at.sidebar.radio[0].set_value("Org-AI-R Calculator").run()

    # Provide invalid dimension scores (non-float string)
    at.text_area[0].set_value("70.0, invalid, 68.0").run()

    # Click the calculate button
    at.form[0].button[0].click().run()

    # Verify error message is displayed and no result is stored in session state
    assert at.error[0].value.startswith("Error calculating Org-AI-R score:")
    assert at.session_state.org_air_result is None


def test_sidebar_navigation_to_scenario_analysis():
    """
    Test navigation to the 'Scenario Analysis & Visualization' page and verify initial content.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Select "Scenario Analysis & Visualization" in the sidebar radio button
    at.sidebar.radio[0].set_value("Scenario Analysis & Visualization").run()

    # Verify the title and introductory markdowns of the scenario analysis page
    assert at.markdown[0].value == "# Org-AI-R Scenario Analysis"
    assert "Visualizing Org-AI-R Scores and Confidence Intervals Across Scenarios" in at.markdown[1].value
    # Check that the "Run Scenario Analysis" button is present but results are not yet shown
    assert at.button[0].label == "Run Scenario Analysis for Multiple Companies"
    assert len(at.dataframe) == 0  # No dataframe yet
    assert len(at.pyplot) == 0  # No plots yet


def test_scenario_analysis_run_button_and_results_display():
    """
    Test the Scenario Analysis page:
    - Clicking the "Run Scenario Analysis" button
    - Verifying the aggregated results DataFrame and plots are displayed.
    """
    at = AppTest.from_file(APP_FILE).run()

    # Navigate to the scenario analysis page
    at.sidebar.radio[0].set_value("Scenario Analysis & Visualization").run()

    # Click the "Run Scenario Analysis for Multiple Companies" button
    at.button[0].click().run()

    # Verify that the aggregated results dataframe is displayed
    assert at.markdown[2].value == "### Aggregated Results Across Scenarios"
    assert at.dataframe[0] is not None
    assert "GLOBAL_LEADER" in at.dataframe[0].value.to_string() # Check for a known company ID in the dataframe's string representation

    # Verify that the first plot (Org-AI-R Scores with CI) is displayed
    assert at.markdown[3].value == "### 2. Org-AI-R Scores with Confidence Intervals"
    assert at.pyplot[0] is not None  # Check for the presence of the plot object

    # Verify that the second plot (Reliability vs. Evidence Count) is displayed
    assert at.markdown[4].value == "### 3. Reliability vs. Evidence Count (Spearman-Brown Prophecy)"
    assert at.pyplot[1] is not None  # Check for the presence of the plot object

    # Verify that all_scenario_results is populated in session state
    assert at.session_state["all_scenario_results"] is not None
    assert len(at.session_state["all_scenario_results"]) == 4  # Expecting 4 scenarios


def test_scenario_analysis_results_persistence_on_rerun():
    """
    Test that scenario analysis results persist in session state
    and are displayed immediately on revisiting the page without
    re-clicking the run button, if already calculated.
    """
    at = AppTest.from_file(APP_FILE).run()

    # 1. Navigate to scenario page and run analysis
    at.sidebar.radio[0].set_value("Scenario Analysis & Visualization").run()
    at.button[0].click().run()
    assert at.session_state.all_scenario_results is not None
    assert len(at.dataframe) == 1
    assert len(at.pyplot) == 2

    # 2. Simulate a full rerun, by passing the existing session state to a new AppTest instance.
    # This simulates a browser refresh where Streamlit preserves the session_state.
    at_rerun = AppTest.from_file(APP_FILE)
    at_rerun.session_state = at.session_state  # Explicitly carry over the session state
    at_rerun.run()

    # Verify that the current_page is still the scenario analysis page
    assert at_rerun.session_state.current_page == "Scenario Analysis & Visualization"

    # Verify that results are still displayed without needing to click the button again
    assert at_rerun.dataframe[0] is not None
    assert at_rerun.pyplot[0] is not None
    assert at_rerun.pyplot[1] is not None
    assert at_rerun.session_state.all_scenario_results is not None
    assert at_rerun.button[0].label == "Run Scenario Analysis for Multiple Companies"  # Button is still there


def test_calculator_input_param_preservation_on_rerun():
    """
    Test that input parameters in the calculator form are preserved
    in session state across reruns.
    """
    at = AppTest.from_file(APP_FILE).run()

    # 1. Navigate to calculator page and change some inputs
    at.sidebar.radio[0].set_value("Org-AI-R Calculator").run()
    at.text_input[0].set_value("MODIFIED_CORP").run()
    at.number_input[0].set_value(50.0).run()  # Talent Concentration
    at.number_input[5].set_value(30).run()    # Evidence Count

    # 2. Simulate a full rerun, ensuring session_state is passed
    at_rerun = AppTest.from_file(APP_FILE)
    at_rerun.session_state = at.session_state  # Explicitly carry over the session state
    at_rerun.run()

    # 3. Navigate back to the calculator page on the rerun instance
    at_rerun.sidebar.radio[0].set_value("Org-AI-R Calculator").run()

    # Verify that the modified values are still present in the widgets
    assert at_rerun.text_input[0].value == "MODIFIED_CORP"
    assert at_rerun.number_input[0].value == 50.0
    assert at_rerun.number_input[5].value == 30
    assert at_rerun.session_state.input_params["company_id"] == "MODIFIED_CORP"
