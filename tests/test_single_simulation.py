import argparse
from datetime import datetime
import subprocess
import sys

import numpy as np

from sch_simulation.helsim_FUNC_KK.file_parsing import (
    parse_coverage_input,
    readCoverageFile,
    parse_vector_control_input,
)
from sch_simulation.helsim_FUNC_KK.configuration import setupSD
from sch_simulation.helsim_RUN_KK import (
    loadParameters, doRealizationSurveyCoveragePickle,
)

SEED = 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-U", "--update-ref", action="store_true")
    return parser


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


def check_prevalence_values(values: np.ndarray) -> int:
    REF_FILEPATH = "tests/data/prevalence_reference.txt"
    reference_values = np.loadtxt(REF_FILEPATH)
    try:
        np.testing.assert_allclose(values, reference_values)
    except AssertionError:
        return 1
    return 0


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    np.random.seed(seed=SEED)
    params = setup_parameters()
    results, _ = doRealizationSurveyCoveragePickle(
        params=params, surveyType="KK2", simData=setupSD(params), mult=5,
    )
    values = np.array([res.prevalence for res in results])
    if args.update_ref:
        o = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            encoding="ASCII",
        )
        header = (
            f"Generated {sys.argv[0]} on {datetime.today()}\n"
            f"Revision no {o.stdout}\n"
            f"seed = {SEED}"
        )
        np.savetxt(
            "tests/data/prevalence_reference.txt",
            values,
            fmt="%5.3f",
            header=header,
        )
    else:
        errcode = check_prevalence_values(values)
        sys.exit(errcode)
