from dataclasses import dataclass
from functools import cache
from typing import Literal
import pandas
import sch_simulation
import numpy as np
from sch_simulation.helsim_FUNC_KK.configuration import setupSD
from sch_simulation.helsim_FUNC_KK.file_parsing import (
    parse_coverage_input,
    parse_vector_control_input,
    readCoverageFile,
)

import sch_simulation.helsim_RUN_KK

import sch_simulation.helsim_FUNC_KK.results_processing as results_processing


@dataclass(eq=True, frozen=True)
class FixedParameters:
    number_hosts: int
    """Number of people to model

    The higher the value of N, the more consistent the results will be
    though the longer the simulation will take."""

    coverage_file_name: str
    demography_name: str
    """Which demography to use"""

    survey_type: Literal["KK1", "KK2", "POC-CCA", "PCR"]
    """Which suvery type to use
    
    Must be one KK1, KK2, POC-CCA or PCR"""

    coverage_text_file_storage_name: str
    """File name to store coverage information in"""

    parameter_file_name: str
    """Standard parameter file path (in sch_simulation/data folder)"""

    min_multiplier: int
    """Used to speed up running the model
    
    A higher number will result in faster but less accurate simulation."""


@cache
def returnYearlyPrevalenceEstimate(R0, k, fixed_parameters: FixedParameters):
    # Run `parse_coverage_input` for side-effect of writing coverage
    # text file.
    parse_coverage_input(
        fixed_parameters.coverage_file_name,
        fixed_parameters.coverage_text_file_storage_name,
    )
    # initialize the parameters
    params = sch_simulation.helsim_RUN_KK.loadParameters(
        fixed_parameters.parameter_file_name, fixed_parameters.demography_name
    )
    # add coverage data to parameters file
    params = readCoverageFile(fixed_parameters.coverage_text_file_storage_name, params)
    # add vector control data to parameters
    params = parse_vector_control_input(fixed_parameters.coverage_file_name, params)

    # update R0, k and population size as desired
    params.R0 = R0
    params.k = k
    params.N = fixed_parameters.number_hosts

    # setup the initial simData

    simData = setupSD(params)

    # run a single realization
    results, SD = sch_simulation.helsim_RUN_KK.doRealizationSurveyCoveragePickle(
        params,
        fixed_parameters.survey_type,
        simData,
        fixed_parameters.min_multiplier,
    )

    # process the output
    output = results_processing.extractHostData([results])

    # do a single simulation
    numReps = 1
    PrevalenceEstimate = results_processing.getPrevalenceWholePop(
        output, params, numReps, params.Unfertilized, fixed_parameters.survey_type, 1
    )
    return PrevalenceEstimate


def extract_relevant_results(results: pandas.DataFrame) -> float:
    num_years = len(results)
    # TODO: for now we assume we are interested in fitting the last year
    return results[results_processing.OUTPUT_COLUMN_NAME][num_years - 1]


def run_model_with_parameters(seeds, parameters, fixed_parameters: FixedParameters):
    if len(seeds) != len(parameters):
        raise ValueError(
            f"Must have same number of seeds as parameters {len(seeds)} != {len(parameters)}"
        )

    num_runs = len(seeds)

    final_prevalence_for_each_run = []

    for seed, parameter_set in zip(seeds, parameters):
        R0 = parameter_set[0]
        k = parameter_set[1]

        results = returnYearlyPrevalenceEstimate(R0, k, fixed_parameters)

        prevalence = extract_relevant_results(results)
        final_prevalence_for_each_run.append(prevalence)

    results_np_array = np.array(final_prevalence_for_each_run).reshape(num_runs, 1)
    return results_np_array
