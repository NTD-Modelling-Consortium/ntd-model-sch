import pytest
from sch_simulation.amis_integration.amis_integration import extract_relevant_results
import pandas as pd
from pandas import testing as pdt


def test_extract_data():
    example_results = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "draw_1": [0.1, 0.2, 0.3]})
    outcome = extract_relevant_results(example_results, [2.0])
    pdt.assert_series_equal(
        outcome, pd.Series(data=[0.3], index=[2.0], name="Prevalence")
    )


def test_extract_multiple_years():
    example_results = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "draw_1": [0.1, 0.2, 0.3]})
    outcome = extract_relevant_results(example_results, [0.0, 2.0])
    pdt.assert_series_equal(
        outcome, pd.Series(data=[0.1, 0.3], index=[0.0, 2.0], name="Prevalence")
    )


def test_extract_missing_year():
    example_results = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "draw_1": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        outcome = extract_relevant_results(example_results, [3.0])
