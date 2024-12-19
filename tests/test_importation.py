import argparse
from datetime import datetime
import subprocess
import sys
import numpy.testing as npt
import unittest  # Import unittest

import numpy as np

from sch_simulation.helsim_FUNC_KK.file_parsing import *
from sch_simulation.helsim_FUNC_KK.configuration import setupSD
from sch_simulation.helsim_RUN_KK import (
    loadParameters
)
from sch_simulation.helsim_FUNC_KK.events import doImportation

SEED = 1


def setup_parameters():
    # Writes 'coverageTextFileStorageName' on disk
    parse_coverage_input(
        coverageFileName="mansoni_coverage_scenario_0.xlsx",
        coverageTextFileStorageName="Man_MDA_vacc.txt",
    )

    params = loadParameters(
        paramFileName="mansoni_params.txt", demogName="UgandaRural",
    )
    params = readCoverageFile(
        coverageTextFileStorageName="Man_MDA_vacc.txt", params=params,
    )
    params = parse_vector_control_input(
        coverageFileName="mansoni_coverage_scenario_0.xlsx", params=params,
    )
    params.R0 = 3
    params.k = 0.04
    params.N = 100

    return params

def test_importation():
        np.random.seed(seed=SEED)
        params = setup_parameters()
        simData = setupSD(params)
        import_indivs = [0,1]
        simData.worms.female[import_indivs] = [0,0]
        simData = doImportation(simData, import_indivs, params, 0)
        npt.assert_array_almost_equal(simData.worms.female[import_indivs], [34,448])
