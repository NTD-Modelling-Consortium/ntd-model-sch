import copy
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pkg_resources
from numpy.typing import NDArray

from sch_simulation.helsim_FUNC_KK.helsim_structures import Coverage, Parameters

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")


def params_from_contents(contents: List[str]) -> Dict[str, Any]:
    params = {}
    for content in contents:
        line = content.split("\t")
        if len(line) >= 2:
            key = line[0]
            try:
                float_converted_list = [float(x) for x in line[1].split(" ")]
            except ValueError:
                float_converted_list = None

            if float_converted_list is None:
                new_v = line[1]
            elif len(float_converted_list) == 0:
                raise ValueError(f"No values supplied in {key}")
            elif len(float_converted_list) == 1:
                new_v = float_converted_list[0]
            else:
                new_v = np.array(float_converted_list)

            params[key] = new_v
    return params


def readParam(fileName: str) -> Dict[str, Any]:

    """
    This function extracts the parameter values stored
    in the input text files into a dictionary.
    Parameters
    ----------
    fileName: str
        name of the input text file;
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    """

    DATA_PATH = pkg_resources.resource_filename("sch_simulation", "data/")

    with open(DATA_PATH + fileName) as f:
        contents = f.readlines()

    return params_from_contents(contents)


def readCovFile(fileName: str) -> Dict[str, Any]:

    """
    This function extracts the parameter values stored
    in the input text files into a dictionary.
    Parameters
    ----------
    fileName: str
        name of the input text file; this should be a
        relative or absolute filesystem path provided
        by the caller of the library functions
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    """

    with open(fileName) as f:
        contents = f.readlines()

    return params_from_contents(contents)


def parse_coverage_input(
    coverageFileName: str, coverageTextFileStorageName: str
) -> str:
    """
    This function extracts the coverage data and stores in a text file

    Parameters
    ----------
    coverageFileName: str
        name of the input text file;
    coverageTextFileStorageName: str
        name of txt file in which to store processed intervention data
    Returns
    -------
    coverageText: str
        string variable holding all coverage information for given file name;
    """

    # read in Coverage spreadsheet
    DATA_PATH = pkg_resources.resource_filename("sch_simulation", "data/")
    PlatCov = pd.read_excel(
        DATA_PATH + coverageFileName, sheet_name="Platform Coverage"
    )
    # which rows are for MDA and vaccine
    intervention_array = PlatCov["Intervention Type"]
    MDARows = np.where(np.array(intervention_array == "Treatment"))[0]
    VaccRows = np.where(np.array(intervention_array == "Vaccine"))[0]

    # initialize variables to contain Age ranges, years and coverage values for MDA and vaccine
    MDAAges = np.zeros([len(MDARows), 2])
    MDAYears = []
    MDACoverages = []
    VaccAges = np.zeros([len(VaccRows), 2])
    VaccYears = []
    VaccCoverages = []

    # we want to find which is the first year specified in the coverage data, along with which
    # column of the data set this corresponds to
    fy = 10000
    fy_index = 10000
    for i in range(len(PlatCov.columns)):
        if type(PlatCov.columns[i]) == int:
            fy = min(fy, PlatCov.columns[i])
            fy_index = min(fy_index, i)

    include = []
    for i in range(len(MDARows)):
        k = MDARows[i]
        w = PlatCov.iloc[k, :]
        greaterThan0 = 0
        for j in range(fy_index, len(PlatCov.columns)):
            cname = PlatCov.columns[j]
            if w[cname] > 0:
                greaterThan0 += 1
        if greaterThan0 > 0:
            include.append(k)
    MDARows = include

    include = []
    for i in range(len(VaccRows)):
        k = VaccRows[i]
        w = PlatCov.iloc[k, :]
        greaterThan0 = 0
        for j in range(fy_index, len(PlatCov.columns)):
            cname = PlatCov.columns[j]
            if w[cname] > 0:
                greaterThan0 += 1
        if greaterThan0 > 0:
            include.append(k)

    VaccRows = include

    # store number of age ranges specified for MDA coverage
    numMDAAges = len(MDARows)
    # initialize MDA text storage with the number of age groups specified for MDA
    MDA_txt = "nMDAAges" + "\t" + str(numMDAAges) + "\n"
    # add drug efficiencies for 2 MDNA drugs
    MDA_txt = (
        MDA_txt + "drug1Eff\t" + str(0.87) + "\n" + "drug2Eff\t" + str(0.95) + "\n"
    )

    # store number of age ranges specified for Vaccine coverage
    numVaccAges = len(VaccRows)
    # initialize vaccine text storage with the number of age groups specified for vaccination
    Vacc_txt = "nVaccAges" + "\t" + str(numVaccAges) + "\n"

    # loop over MDA coverage rows
    for i in range(len(MDARows)):
        # get row number of each MDA entry
        k = MDARows[i]
        # store this row
        w = PlatCov.iloc[int(k), :]
        # store the min and maximum age of this MDA row
        MDAAges[i, :] = np.array([w["min age"], w["max age"]])
        # re initilize the coverage and years data
        MDAYears = []
        MDACoverages = []
        # loop over the yearly data for this row
        for j in range(fy_index, len(PlatCov.columns)):
            # get the column name of specified column
            cname = PlatCov.columns[j]
            # if the coverage is >0, then add the year and coverage to the appropriate variable
            if w[cname] > 0:
                MDAYears.append(cname)
                MDACoverages.append(w[cname])

        MDA_txt = (
            MDA_txt
            + "MDA_age"
            + str(i + 1)
            + "\t"
            + str(int(MDAAges[i, :][0]))
            + " "
            + str(int(MDAAges[i, :][1]))
            + "\n"
        )
        MDA_txt = MDA_txt + "MDA_Years" + str(i + 1) + "\t"
        for k in range(len(MDAYears)):
            if k == (len(MDAYears) - 1):
                MDA_txt = MDA_txt + str(MDAYears[k]) + "\n"
            else:
                MDA_txt = MDA_txt + str(MDAYears[k]) + " "
        MDA_txt = MDA_txt + "MDA_Coverage" + str(i + 1) + "\t"
        for k in range(len(MDACoverages)):
            if k == (len(MDACoverages) - 1):
                MDA_txt = MDA_txt + str(MDACoverages[k]) + "\n"
            else:
                MDA_txt = MDA_txt + str(MDACoverages[k]) + " "

    # loop over Vaccination coverage rows
    for i in range(len(VaccRows)):
        # get row number of each MDA entry
        k = VaccRows[i]
        # store this row
        w = PlatCov.iloc[int(k), :]
        # store the min and maximum age of this Vaccine row
        VaccAges[i, :] = np.array([w["min age"], w["max age"]])
        # re initilize the coverage and years data
        VaccYears = []
        VaccCoverages = []
        # loop over the yearly data for this row
        for j in range(fy_index, len(PlatCov.columns)):
            # get the column name of specified column
            cname = PlatCov.columns[j]
            # if coverage is >0 then add the year and coverage to the appropriate variable
            if w[cname] > 0:
                VaccYears.append(cname)
                VaccCoverages.append(w[cname])
        # once all years and coverages have been collected, we store these in a string variable
        Vacc_txt = (
            Vacc_txt
            + "Vacc_age"
            + str(i + 1)
            + "\t"
            + str(int(VaccAges[i, :][0]))
            + " "
            + str(int(VaccAges[i, :][1]))
            + "\n"
        )
        Vacc_txt = Vacc_txt + "Vacc_Years" + str(i + 1) + "\t"
        for k in range(len(VaccYears)):
            if k == (len(VaccYears) - 1):
                Vacc_txt = Vacc_txt + str(VaccYears[k]) + "\n"
            else:
                Vacc_txt = Vacc_txt + str(VaccYears[k]) + " "
        Vacc_txt = Vacc_txt + "Vacc_Coverage" + str(i + 1) + "\t"
        for k in range(len(VaccCoverages)):
            if k == (len(VaccCoverages) - 1):
                Vacc_txt = Vacc_txt + str(VaccCoverages[k]) + "\n"
            else:
                Vacc_txt = Vacc_txt + str(VaccCoverages[k]) + " "

    # read in market share data
    MarketShare = pd.read_excel(DATA_PATH + coverageFileName, sheet_name="MarketShare")
    # find which rows store data for MDAs
    MDAMarketShare = np.where(np.array(MarketShare["Platform"] == "MDA"))[0]
    # initialize variable to store which drug is being used
    MDASplit = np.zeros(len(MDAMarketShare))
    # find which row holds data for the Old and New drugs
    # these will be stored at 1 and 2 respectively
    for i in range(len(MDAMarketShare)):
        if "Old" in MarketShare["Product"][int(MDAMarketShare[i])]:
            MDASplit[i] = 1
        else:
            MDASplit[i] = 2

    # we want to find which is the first year specified in the coverage data, along with which
    # column of the data set this corresponds to
    fy = 10000
    fy_index = 10000
    for i in range(len(MarketShare.columns)):
        if type(MarketShare.columns[i]) == int:
            fy = min(fy, MarketShare.columns[i])
            fy_index = min(fy_index, i)

    # loop over Market share MDA rows
    for i in range(len(MDAMarketShare)):
        # store which row we are on
        k = MDAMarketShare[i]
        # get data for this row
        w = MarketShare.iloc[int(k), :]
        # initialize needed arrays
        MDAYears = []
        MDAYearSplit = []
        drugInd = MDASplit[i]
        # loop over yearly market share data
        for j in range(fy_index, len(MarketShare.columns)):
            # get column name for this column
            cname = MarketShare.columns[j]
            # if split is >0 then store the year and split in appropriate variables
            if w[cname] > 0:
                MDAYears.append(cname)
                MDAYearSplit.append(w[cname])
        if len(MDAYears) == 0:
            MDAYears.append(10000)
            MDAYearSplit.append(0)
        # once we have looped over each year, we store add this information to the MDA string variable
        MDA_txt = MDA_txt + "drug" + str(int(drugInd)) + "Years\t"
        for k in range(len(MDAYears)):
            if k == (len(MDAYears) - 1):
                MDA_txt = MDA_txt + str(MDAYears[k]) + "\n"
            else:
                MDA_txt = MDA_txt + str(MDAYears[k]) + " "

        MDA_txt = MDA_txt + "drug" + str(int(drugInd)) + "Split\t"
        for k in range(len(MDAYearSplit)):
            if k == (len(MDAYearSplit) - 1):
                MDA_txt = MDA_txt + str(MDAYearSplit[k]) + "\n"
            else:
                MDA_txt = MDA_txt + str(MDAYearSplit[k]) + " "

    coverageText = MDA_txt + Vacc_txt
    # store the Coverage data in a text file
    with open(coverageTextFileStorageName, "w", encoding="utf-8") as f:
        f.write(coverageText)

    return coverageText


def nextMDAVaccInfo(
    params: Parameters,
) -> Tuple[Dict, Dict, int, List[int], List[int], int, List[int], List[int]]:
    chemoTiming = {}
    assert params.MDA is not None
    assert params.Vacc is not None

    for i, mda in enumerate(params.MDA):
        chemoTiming["Age{0}".format(i)] = copy.deepcopy(mda.Years)
    VaccTiming = {}
    for i, vacc in enumerate(params.Vacc):
        VaccTiming["Age{0}".format(i)] = copy.deepcopy(vacc.Years)
    #  currentVaccineTimings = copy.deepcopy(params['VaccineTimings'])

    nextChemoTime = 10000
    for i, mda in enumerate(params.MDA):
        nextChemoTime = min(nextChemoTime, min(chemoTiming["Age{0}".format(i)]))
    nextMDAAge = []
    for i, mda in enumerate(params.MDA):
        if nextChemoTime == min(chemoTiming["Age{0}".format(i)]):
            nextMDAAge.append(i)
    nextChemoIndex = []
    for i in range(len(nextMDAAge)):
        k = nextMDAAge[i]
        nextChemoIndex.append(np.argmin(np.array(chemoTiming["Age{0}".format(k)])))

    nextVaccTime = 10000
    for i, vacc in enumerate(params.Vacc):
        nextVaccTime = min(nextVaccTime, min(VaccTiming["Age{0}".format(i)]))
    nextVaccAge = []
    for i, vacc in enumerate(params.Vacc):
        if nextVaccTime == min(VaccTiming["Age{0}".format(i)]):
            nextVaccAge.append(i)
    nextVaccIndex = []
    for i in range(len(nextVaccAge)):
        k = nextVaccAge[i]
        nextVaccIndex.append(np.argmin(np.array(VaccTiming["Age{0}".format(k)])))

    return (
        chemoTiming,
        VaccTiming,
        nextChemoTime,
        nextMDAAge,
        nextChemoIndex,
        nextVaccTime,
        nextVaccAge,
        nextVaccIndex,
    )


def overWritePostVacc(
    params: Parameters,
    nextVaccAge: Union[NDArray[np.int_], List[int]],
    nextVaccIndex: Union[NDArray[np.int_], List[int]],
):
    assert params.Vacc is not None
    for i in range(len(nextVaccAge)):
        k = nextVaccIndex[i]
        j = nextVaccAge[i]
        params.Vacc[j].Years[k] = 10000

    return params


def overWritePostMDA(
    params: Parameters,
    nextMDAAge: Union[NDArray[np.int_], List[int]],
    nextChemoIndex: Union[NDArray[np.int_], List[int]],
):
    assert params.MDA is not None
    for i in range(len(nextMDAAge)):
        k = nextChemoIndex[i]
        j = nextMDAAge[i]
        params.MDA[j].Years[k] = 10000

    return params


def readCoverageFile(
    coverageTextFileStorageName: str, params: Parameters
) -> Parameters:

    coverage = readCovFile(coverageTextFileStorageName)

    nMDAAges = int(coverage["nMDAAges"])
    nVaccAges = int(coverage["nVaccAges"])
    mda_covs = []
    for i in range(nMDAAges):
        cov = Coverage(
            Age=coverage["MDA_age" + str(i + 1)],
            Years=coverage["MDA_Years" + str(i + 1)] - 2018,
            Coverage=coverage["MDA_Coverage" + str(i + 1)],
        )
        mda_covs.append(cov)
    params.MDA = mda_covs
    vacc_covs = []
    for i in range(nVaccAges):
        cov = Coverage(
            Age=coverage["Vacc_age" + str(i + 1)],
            Years=coverage["Vacc_Years" + str(i + 1)] - 2018,
            Coverage=coverage["Vacc_Coverage" + str(i + 1)],
        )
        vacc_covs.append(cov)
    params.Vacc = mda_covs
    params.drug1Years = np.array(coverage["drug1Years"] - 2018)
    params.drug1Split = np.array(coverage["drug1Split"])
    params.drug2Years = np.array(coverage["drug2Years"] - 2018)
    params.drug2Split = np.array(coverage["drug2Split"])
    return params


def readParams(
    paramFileName: str,
    demogFileName: str = "Demographies.txt",
    demogName: str = "Default",
) -> Parameters:

    """
    This function organizes the model parameters and
    the demography parameters into a unique dictionary.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogFileName: str
        name of the input text file with the demography parameters;
    demogName: str
        subset of demography parameters to be extracted;
    Returns
    -------
    params: Parameters
        dataclass containing the parameter names and values;
    """

    demographies = readParam(demogFileName)
    parameters = readParam(paramFileName)
    chemoTimings1 = np.array(
        [
            float(parameters["treatStart1"] + x * parameters["treatInterval1"])
            for x in range(int(parameters["nRounds1"]))
        ]
    )

    chemoTimings2 = np.array(
        [
            parameters["treatStart2"] + x * parameters["treatInterval2"]
            for x in range(int(parameters["nRounds2"]))
        ]
    )

    VaccineTimings = np.array(
        [
            parameters["VaccTreatStart"] + x * parameters["treatIntervalVacc"]
            for x in range(int(parameters["nRoundsVacc"]))
        ]
    )

    params = Parameters(
        numReps=int(parameters["repNum"]),
        maxTime=parameters["nYears"],
        N=int(parameters["nHosts"]),
        R0=parameters["R0"],
        lambda_egg=parameters["lambda"],
        v2=parameters["v2lambda"],
        gamma=parameters["gamma"],
        k=parameters["k"],
        sigma=parameters["sigma"],
        v1=parameters["v1sigma"],
        LDecayRate=parameters["ReservoirDecayRate"],
        DrugEfficacy=parameters["drugEff"],
        DrugEfficacy1=parameters["drugEff1"],
        DrugEfficacy2=parameters["drugEff2"],
        contactAgeBreaks=parameters["contactAgeBreaks"],
        contactRates=parameters["betaValues"],
        v3=parameters["v3betaValues"],
        rho=parameters["rhoValues"],
        treatmentAgeBreaks=parameters["treatmentBreaks"],
        VaccTreatmentBreaks=parameters["VaccTreatmentBreaks"],
        coverage1=parameters["coverage1"],
        coverage2=parameters["coverage2"],
        VaccCoverage=parameters["VaccCoverage"],
        treatInterval1=parameters["treatInterval1"],
        treatInterval2=parameters["treatInterval2"],
        treatStart1=parameters["treatStart1"],
        treatStart2=parameters["treatStart2"],
        nRounds1=int(parameters["nRounds1"]),
        nRounds2=int(parameters["nRounds2"]),
        chemoTimings1=chemoTimings1,
        chemoTimings2=chemoTimings2,
        VaccineTimings=VaccineTimings,
        outTimings=parameters["outputEvents"],
        propNeverCompliers=parameters["neverTreated"],
        highBurdenBreaks=parameters["highBurdenBreaks"],
        highBurdenValues=parameters["highBurdenValues"],
        VaccDecayRate=parameters["VaccDecayRate"],
        VaccTreatStart=parameters["VaccTreatStart"],
        nRoundsVacc=parameters["nRoundsVacc"],
        treatIntervalVacc=parameters["treatIntervalVacc"],
        heavyThreshold=parameters["heavyThreshold"],
        mediumThreshold=parameters["mediumThreshold"],
        sampleSizeOne=int(parameters["sampleSizeOne"]),
        sampleSizeTwo=int(parameters["sampleSizeTwo"]),
        nSamples=int(parameters["nSamples"]),
        minSurveyAge=parameters["minSurveyAge"],
        maxSurveyAge=parameters["maxSurveyAge"],
        demogType=demogName,
        hostMuData=demographies[demogName + "_hostMuData"],
        muBreaks=np.append(0, demographies[demogName + "_upperBoundData"]),
        SR=parameters["StochSR"] == "TRUE",
        reproFuncName=parameters["reproFuncName"],
        z=np.exp(-parameters["gamma"]),
        k_epg=parameters["k_epg"],
        species=parameters["species"],
        timeToFirstSurvey=parameters["timeToFirstSurvey"],
        timeToNextSurvey=parameters["timeToNextSurvey"],
        surveyThreshold=parameters["surveyThreshold"],
        Unfertilized=parameters["unfertilized"],
    )

    return params
