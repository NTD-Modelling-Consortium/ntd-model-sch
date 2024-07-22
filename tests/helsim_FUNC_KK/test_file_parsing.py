import pytest
import sch_simulation
from sch_simulation.amis_integration.amis_integration import extract_relevant_results, returnYearlyPrevalenceEstimate, FixedParameters, run_model_with_parameters
import pandas as pd
from pandas import testing as pdt
from numpy import testing as npt

from sch_simulation.helsim_FUNC_KK.file_parsing import parse_coverage_input
from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters

example_parameters = FixedParameters(
        # the higher the value of N, the more consistent the results will be
        # though the longer the simulation will take
        number_hosts = 10,
        # no intervention
        coverage_file_name = "mansoni_coverage_scenario_0.xlsx",
        demography_name = "UgandaRural",
        # cset the survey type to Kato Katz with duplicate slide
        survey_type = "KK2",
        parameter_file_name = "mansoni_params.txt",
        coverage_text_file_storage_name = "Man_MDA_vacc.txt",
        # the following number dictates the number of events (e.g. worm deaths)
        # we allow to happen before updating other parts of the model
        # the higher this number the faster the simulation
        # (though there is a check so that there can't be too many events at once)
        # the higher the number the greater the potential for
        # errors in the model accruing.
        # 5 is a reasonable level of compromise for speed and errors, but using
        # a lower value such as 3 is also quite good
        min_multiplier = 5
    )

def test_valid_input():

    input_file = 'test_inputs/InputMDA_MTP_7.xlsx'
    output_file = 'cov_file_test.txt'

    parse_coverage_input(
        input_file,
        output_file
    )

    params = sch_simulation.helsim_RUN_KK.loadParameters(
        example_parameters.parameter_file_name, example_parameters.demography_name
    )
    # add coverage data to parameters file
    params = sch_simulation.helsim_FUNC_KK.readCoverageFile(output_file, params)

    
    npt.assert_array_equal(params.drug2Split, [ 1.0, 1.0, 1.0])
    assert params.drug2Split[0] == 1.0

def test_coverage_file_where_drug2_has_only_one_year():

    input_file = 'test_inputs/InputMDA_MTP_8.xlsx'
    output_file = 'cov_file_inv_test.txt'

    parse_coverage_input(
        input_file,
        output_file
    )

    params = sch_simulation.helsim_RUN_KK.loadParameters(
        example_parameters.parameter_file_name, example_parameters.demography_name
    )
    # add coverage data to parameters file
    params = sch_simulation.helsim_FUNC_KK.readCoverageFile(output_file, params)

    assert params.drug2Split[0] == 1.0
    npt.assert_array_equal(params.drug2Split, [ 1.0 ])

def test_coverage_file_with_single_coverage_value_for_mda():

    input_file = 'test_inputs/InputMDA_MTP_1378.xlsx'
    output_file = 'cov_file_bb_test.txt'

    parse_coverage_input(
        input_file,
        output_file
    )

    params = sch_simulation.helsim_RUN_KK.loadParameters(
        example_parameters.parameter_file_name, example_parameters.demography_name
    )
    # add coverage data to parameters file
    params = sch_simulation.helsim_FUNC_KK.readCoverageFile(output_file, params)

    assert len(params.MDA[0].Coverage) == 1