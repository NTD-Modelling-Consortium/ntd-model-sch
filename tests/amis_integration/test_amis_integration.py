from sch_simulation.amis_integration.amis_integration import extract_relevant_results
import pandas as pd


def test_extract_data():
    example_results = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "draw_1": [0.1, 0.2, 0.3]})

    assert extract_relevant_results(example_results) == 0.3
