import numpy as np

import pandas as pd
from scipy.optimize import bisect
import warnings
import copy
import random
import pkg_resources
from dataclasses import dataclass
from scipy.special import gamma
warnings.filterwarnings('ignore')

np.seterr(divide='ignore')

import sch_simulation.ParallelFuncs as ParallelFuncs
from numpy import ndarray
from typing import Callable, List, Tuple, Optional, Dict
from numpy.typing import NDArray
@dataclass
class MonogParameters:
    c_k: float
    cosTheta: ndarray

@dataclass
class Equilibrium:
    stableProfile: ndarray
    ageValues: ndarray
    L_stable: float
    L_breakpoint: float
    K_values: ndarray
    L_values: ndarray
    FOIMultiplier: float
    hostSurvival: Optional[ndarray] = None

@dataclass
class Coverage:
    Age: ndarray
    Years: ndarray # 1-D array lower/upper
    Coverage: ndarray # 1-D array

@dataclass
class Parameters:
    numReps: int
    maxTime: float #nYears
    N: int #nHosts
    R0: float # Basic reproductive number
    lambda_egg: float # Eggs per gram
    v2: NDArray[np.float_] # Fraction of eggs produced when vaccinated.
    gamma: float # Exponential density dependence of parasite adult stage
    k: float # Shape parameter of assumed negative binomial distribution of worms amongst host
    sigma: float # Worm death rate
    v1: NDArray[np.float_] # impact of vaccine on worm death rate KK. Assume worm death rate is v1*sigma.
    LDecayRate: float # ReservoirDecayRate
    DrugEfficacy: float
    DrugEfficacy1: float
    DrugEfficacy2: float
    contactAgeBreaks: ndarray # 1-D array - Contact age group breaks (minus sign necessary to include zero age)
    contactRates: ndarray # 1-D array - BetaValues: Relative contact rates
    v3: ndarray # 1-D array, v3 beta values: impact of vaccine on contact rates  Assume contact rate under vaccination is times v3. KK 
    rho: ndarray # 1-D array, - Rho, contribution to the reservoir by contact age group. 
    treatmentAgeBreaks: ndarray # 1-D array, treatmentBreaks Minimum age of each treatment group (minus sign necessary to include zero age): Infants; Pre-SAC; SAC; Adults
    VaccTreatmentBreaks: ndarray # 1-D array, age range of vaccinated group.  ## KK: these are the lower bounds of ranges with width 1. THEY MUST BE > 1 YEAR APART!!
    coverage1: ndarray
    coverage2: ndarray
    VaccCoverage: ndarray # Vaccine coverage of the age groups KK
    #VaccEfficacy
    treatInterval1: int # interval between treatments in years. 
    treatInterval2: int # interval between treatments in years. 
    treatStart1: float
    treatStart2: float
    nRounds1: int
    nRounds2: int
    chemoTimings1: ndarray # 1-D array
    chemoTimings2: ndarray # 1-D array
    VaccineTimings: ndarray # 1-D array
    outTimings: ndarray # 1-D array, outputEvents
    propNeverCompliers: float # neverTreated
    highBurdenBreaks: ndarray # 1-D array Three categories here
    highBurdenValues: ndarray # 1-D array
    VaccDecayRate: ndarray #vacc decay rate. rate of vaccine decay = 1/duration of vaccine   A vector with value 0 in state 1 and the vacc decay rate for state 2. KK.
    VaccTreatStart: float ##Vaccine administration year start KK 
    nRoundsVacc: int ##number of vaccine rounds KK 
    treatIntervalVacc: float #KK
    heavyThreshold: int # The threshold for heavy burden of infection, egg count > heavyThreshold
    mediumThreshold: int # The threshold of medium burden of infection, mediumThreshold <= egg count <= heavyThreshold
    sampleSizeOne: int
    sampleSizeTwo: int
    nSamples: int
    minSurveyAge: float
    maxSurveyAge: float
    demogType: str # demogName: subset of demography parameters to be extracted
    reproFuncName: str # name of function for reproduction (a string).  [Deterministic] ## epgPerPerson   epgFertility	epgMonog
    z: float # np.exp(-'gamma'),
    k_epg: float
    species: str
    timeToFirstSurvey: float
    timeToNextSurvey: float
    surveyThreshold: float
    Unfertilized: bool
    hostMuData: ndarray
    muBreaks: ndarray
    SR: bool
    psi: float = 1.0
    reproFunc: Optional[Callable[[np.ndarray, "Parameters"], np.ndarray]] = None
    maxHostAge: Optional[ndarray] = None
    muAges: Optional[ndarray] = None
    hostMu: Optional[float] = None
    monogParams: Optional[MonogParameters] = None
    equiData: Optional[Equilibrium] = None
    hostSurvivalCurve: Optional[ndarray] = None
    hostAgeCumulDistr: Optional[ndarray] = None
    contactAgeGroupBreaks: Optional[ndarray] = None
    treatmentAgeGroupBreaks: Optional[ndarray] = None
    VaccTreatmentAgeGroupBreaks: Optional[ndarray] = None
    #Coverage
    MDA: Optional[List[Coverage]] = None
    Vacc: Optional[List[Coverage]] = None
    drug1Years: Optional[ndarray] = None
    drug1Split: Optional[ndarray] = None
    drug2Years: Optional[ndarray] = None
    drug2Split: Optional[ndarray] = None


@dataclass
class Demography:
    birthDate: ndarray
    deathDate: ndarray

@dataclass
class Worms:
    total: NDArray[np.int_]
    female: NDArray[np.int_]

@dataclass
class SDEquilibrium:
    si: NDArray[np.float_]
    sv: ndarray
    worms: Worms
    freeLiving: float
    demography: Demography
    contactAgeGroupIndices: ndarray
    treatmentAgeGroupIndices: ndarray
    adherenceFactors: ndarray
    compliers: ndarray
    attendanceRecord: List[ndarray]
    ageAtChemo: List
    adherenceFactorAtChemo: List
    vaccCount: int
    numSurvey: int
    vaccinatedFactors: Optional[ndarray] = None
    VaccTreatmentAgeGroupIndices: Optional[ndarray] = None
    sex_id: Optional[ndarray] = None
    nChemo1: Optional[int] = None
    nChemo2: Optional[int] = None

@dataclass
class Result:
    iteration: int
    time: float
    worms: Worms
    hosts: Demography
    vaccState: ndarray
    freeLiving: float
    adherenceFactors: ndarray
    compliers: ndarray
    sex_id: Optional[ndarray] = None
    nVacc: Optional[int] = None
    nChemo: Optional[int] = None
    nChemo1: Optional[int] = None
    nChemo2: Optional[int] = None
    nSurvey: Optional[int] = None
    surveyPass: Optional[int] = None
    elimination: Optional[int] = None

def readParam(fileName: str):

    '''
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
    '''

    DATA_PATH = pkg_resources.resource_filename('sch_simulation', 'data/')

    with open(DATA_PATH + fileName) as f:
        
        contents = f.readlines()

    params = []

    for content in contents:

        line = content.split('\t')

        if len(line) >= 2:

            try:
                
                line[1] = np.array([float(x) for x in line[1].split(' ')])

                if len(line[1]) == 1:
                    
                    line[1] = line[1][0]

            except:
                
                pass

            params.append(line[:2])

    params = dict(params)

    return params



def readCovFile(fileName: str):

    '''
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
    '''

    with open(fileName) as f:
        
        contents = f.readlines()

    params = []

    for content in contents:

        line = content.split('\t')

        if len(line) >= 2:

            try:
                
                line[1] = np.array([float(x) for x in line[1].split(' ')])

                if len(line[1]) == 1:
                    
                    line[1] = line[1][0]

            except:
                
                pass

            params.append(line[:2])

    params = dict(params)

    return params


def parse_coverage_input(
    coverageFileName: str,
    coverageTextFileStorageName: str
) -> str:
    '''
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
    '''

    # read in Coverage spreadsheet
    DATA_PATH = pkg_resources.resource_filename('sch_simulation', 'data/')
    PlatCov = pd.read_excel(DATA_PATH + coverageFileName, sheet_name = 'Platform Coverage')
    # which rows are for MDA and vaccine
    intervention_array = PlatCov['Intervention Type']
    MDARows = np.where(np.array(intervention_array == "Treatment"))[0]
    VaccRows = np.where(np.array(intervention_array == "Vaccine"))[0]
    
    # initialize variables to contain Age ranges, years and coverage values for MDA and vaccine
    MDAAges = np.zeros([len(MDARows),2])
    MDAYears = []
    MDACoverages = []
    VaccAges = np.zeros([len(VaccRows),2])
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
            if w[cname] > 0 :
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
            if w[cname] > 0 :
                greaterThan0 += 1
        if greaterThan0 > 0:
            include.append(k)
    
    VaccRows = include
    
    # store number of age ranges specified for MDA coverage
    numMDAAges = len(MDARows)
    # initialize MDA text storage with the number of age groups specified for MDA
    MDA_txt = 'nMDAAges' + '\t' + str(numMDAAges) + '\n'
    # add drug efficiencies for 2 MDNA drugs
    MDA_txt = MDA_txt + 'drug1Eff\t' + str(0.87) + '\n' +  'drug2Eff\t' + str(0.95) + '\n'
    
    # store number of age ranges specified for Vaccine coverage
    numVaccAges = len(VaccRows)
    # initialize vaccine text storage with the number of age groups specified for vaccination
    Vacc_txt = 'nVaccAges' + '\t' + str(numVaccAges)+ '\n'
    
    
    # loop over MDA coverage rows
    for i in range(len(MDARows)):
        # get row number of each MDA entry 
        k = MDARows[i]
        # store this row
        w = PlatCov.iloc[int(k), :]
        # store the min and maximum age of this MDA row
        MDAAges[i,:] = np.array([w['min age'], w['max age']])
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
        
        MDA_txt = MDA_txt + 'MDA_age' + str(i+1) + '\t'+ str(int(MDAAges[i,:][0])) +' ' + str(int(MDAAges[i,:][1])) +'\n'
        MDA_txt = MDA_txt + 'MDA_Years' + str(i+1) + '\t'
        for k in range(len(MDAYears)):
            if k == (len(MDAYears)-1):
                MDA_txt = MDA_txt + str(MDAYears[k]) + '\n'
            else:
                MDA_txt = MDA_txt + str(MDAYears[k]) + ' '
        MDA_txt = MDA_txt + 'MDA_Coverage' + str(i+1) + '\t'
        for k in range(len(MDACoverages)):
            if k == (len(MDACoverages)-1):
                MDA_txt = MDA_txt + str(MDACoverages[k]) + '\n'
            else:
                MDA_txt = MDA_txt + str(MDACoverages[k]) + ' '
        
    # loop over Vaccination coverage rows
    for i in range(len(VaccRows)):
        # get row number of each MDA entry 
        k = VaccRows[i]
        # store this row
        w = PlatCov.iloc[int(k), :]
        # store the min and maximum age of this Vaccine row
        VaccAges[i,:] = np.array([w['min age'], w['max age']])
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
        Vacc_txt = Vacc_txt + 'Vacc_age'+str(i+1)+'\t'+ str(int(VaccAges[i,:][0]))+' ' + str(int(VaccAges[i,:][1])) +'\n'
        Vacc_txt = Vacc_txt + 'Vacc_Years' + str(i+1) + '\t'
        for k in range(len(VaccYears)):
            if k == (len(VaccYears)-1):
                Vacc_txt = Vacc_txt + str(VaccYears[k]) +'\n'
            else:
                Vacc_txt = Vacc_txt + str(VaccYears[k]) +' '
        Vacc_txt = Vacc_txt + 'Vacc_Coverage' + str(i+1) + '\t'
        for k in range(len(VaccCoverages)):
            if k == (len(VaccCoverages)-1):
                Vacc_txt = Vacc_txt + str(VaccCoverages[k]) + '\n'
            else:
                Vacc_txt = Vacc_txt + str(VaccCoverages[k]) + ' '

    #read in market share data
    MarketShare = pd.read_excel(DATA_PATH + coverageFileName, sheet_name = 'MarketShare')
    # find which rows store data for MDAs
    MDAMarketShare = np.where(np.array(MarketShare['Platform'] == 'MDA'))[0]
    # initialize variable to store which drug is being used
    MDASplit = np.zeros(len(MDAMarketShare))
    # find which row holds data for the Old and New drugs
    # these will be stored at 1 and 2 respectively
    for i in range(len(MDAMarketShare)):
        if 'Old' in MarketShare['Product'][int(MDAMarketShare[i])]:
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
        MDA_txt = MDA_txt + 'drug' + str(int(drugInd)) + 'Years\t'
        for k in range(len(MDAYears)):
            if k == (len(MDAYears) - 1):
                MDA_txt = MDA_txt + str(MDAYears[k]) +'\n'
            else:
                MDA_txt = MDA_txt + str(MDAYears[k]) +' '

        MDA_txt = MDA_txt + 'drug' + str(int(drugInd)) +'Split\t'
        for k in range(len(MDAYearSplit)):
            if k == (len(MDAYearSplit)-1):
                MDA_txt = MDA_txt + str(MDAYearSplit[k]) + '\n'                
            else:
                MDA_txt = MDA_txt + str(MDAYearSplit[k]) + ' '

        
    coverageText = MDA_txt + Vacc_txt    
    # store the Coverage data in a text file
    with open(coverageTextFileStorageName, 'w', encoding='utf-8') as f:
        f.write(coverageText)
    
    return coverageText



def nextMDAVaccInfo(params: Parameters) -> Tuple[
    Dict,
    Dict,
    int,
    List[int],
    List[int],
    int,
    List[int],
    List[int]
]:
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
        
    return chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex



def overWritePostVacc(params: Parameters,  nextVaccAge: ndarray, nextVaccIndex: ndarray):
    assert params.Vacc is not None
    for i in range(len(nextVaccAge)):
        k = nextVaccIndex[i]
        j = nextVaccAge[i]
        params.Vacc[j].Years[k] = 10000
        
    return params

def overWritePostMDA(params:Parameters,  nextMDAAge: ndarray, nextChemoIndex: ndarray):
    assert params.MDA is not None
    for i in range(len(nextMDAAge)):
        k = nextChemoIndex[i]
        j = nextMDAAge[i]
        params.MDA[j].Years[k] = 10000
        
    return params


def readCoverageFile(coverageTextFileStorageName: str, params: Parameters) -> Parameters:

    coverage = readCovFile(coverageTextFileStorageName)

    nMDAAges = int(coverage['nMDAAges'])
    nVaccAges = int(coverage['nVaccAges'])
    mda_covs = []
    for i in range(nMDAAges):
        cov = Coverage(
            Age = coverage['MDA_age'+ str(i+1)],
            Years = coverage['MDA_Years'+ str(i+1)] - 2018,
            Coverage = coverage['MDA_Coverage'+ str(i+1)]
        )
        mda_covs.append(cov)
    params.MDA = mda_covs
    vacc_covs = []
    for i in range(nVaccAges):
        cov = Coverage(
            Age = coverage['Vacc_age'+ str(i+1)],
            Years = coverage['Vacc_Years'+ str(i+1)] - 2018,
            Coverage = coverage['Vacc_Coverage'+ str(i+1)]
        )
        vacc_covs.append(cov)
    params.Vacc = mda_covs
    params.drug1Years = np.array(coverage['drug1Years'] - 2018)
    params.drug1Split = np.array(coverage['drug1Split'])
    params.drug2Years = np.array(coverage['drug2Years'] - 2018)
    params.drug2Split = np.array(coverage['drug2Split'])
    return params

def readParams(paramFileName, demogFileName='Demographies.txt', demogName='Default') -> Parameters:

    '''
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
    params: dict
        dictionary containing the parameter names and values;
    '''

    demographies = readParam(demogFileName)
    parameters = readParam(paramFileName)
    chemoTimings1 = np.array([float(parameters['treatStart1'] + x * parameters['treatInterval1']) for x in range(int(parameters['nRounds1']))])

    chemoTimings2 = np.array([parameters['treatStart2'] + x * parameters['treatInterval2']
    for x in range(int(parameters['nRounds2']))])
    
    VaccineTimings = np.array([parameters['VaccTreatStart'] + x * parameters['treatIntervalVacc']
    for x in range(int(parameters['nRoundsVacc']))])

    params = Parameters(
        numReps = int(parameters['repNum']),
        maxTime = parameters['nYears'],
        N = int(parameters['nHosts']),
        R0 = parameters['R0'],
        lambda_egg = parameters['lambda'],
        v2 = parameters['v2lambda'],
        gamma = parameters['gamma'],
        k = parameters['k'],
        sigma = parameters['sigma'],
        v1 = parameters['v1sigma'],
        LDecayRate = parameters['ReservoirDecayRate'],
        DrugEfficacy = parameters['drugEff'],
        DrugEfficacy1 = parameters['drugEff1'],
        DrugEfficacy2 = parameters['drugEff2'],
        contactAgeBreaks = parameters['contactAgeBreaks'],
        contactRates = parameters['betaValues'],
        v3 = parameters['v3betaValues'],
        rho = parameters['rhoValues'],
        treatmentAgeBreaks = parameters['treatmentBreaks'],
        VaccTreatmentBreaks = parameters['VaccTreatmentBreaks'],
        coverage1 = parameters['coverage1'],
        coverage2 = parameters['coverage2'],
        VaccCoverage = parameters['VaccCoverage'],
        treatInterval1 = parameters['treatInterval1'],
        treatInterval2 = parameters['treatInterval2'],
        treatStart1 = parameters['treatStart1'],
        treatStart2 = parameters['treatStart2'],
        nRounds1 = int(parameters['nRounds1']),
        nRounds2 = int(parameters['nRounds2']),
        chemoTimings1 = chemoTimings1,
        chemoTimings2 = chemoTimings2,
        VaccineTimings = VaccineTimings,
        outTimings = parameters['outputEvents'],
        propNeverCompliers = parameters['neverTreated'],
        highBurdenBreaks = parameters['highBurdenBreaks'],
        highBurdenValues = parameters['highBurdenValues'],
        VaccDecayRate = parameters['VaccDecayRate'],
        VaccTreatStart = parameters['VaccTreatStart'],
        nRoundsVacc = parameters['nRoundsVacc'],
        treatIntervalVacc = parameters['treatIntervalVacc'],
        heavyThreshold = parameters['heavyThreshold'],
        mediumThreshold = parameters['mediumThreshold'],
        sampleSizeOne = int(parameters['sampleSizeOne']),
        sampleSizeTwo = int(parameters['sampleSizeTwo']),
        nSamples = int(parameters['nSamples']),
        minSurveyAge = parameters['minSurveyAge'],
        maxSurveyAge = parameters['maxSurveyAge'],
        demogType = demogName,
        hostMuData = demographies[demogName + '_hostMuData'],
        muBreaks = np.append(0, demographies[demogName + '_upperBoundData']),
        SR = parameters['StochSR'] == 'TRUE',
        reproFuncName = parameters['reproFuncName'],
        z = np.exp(-parameters['gamma']),
        k_epg = parameters['k_epg'],
        species = parameters['species'],
        timeToFirstSurvey = parameters['timeToFirstSurvey'],
        timeToNextSurvey = parameters['timeToNextSurvey'],
        surveyThreshold = parameters['surveyThreshold'],
        Unfertilized = parameters['unfertilized']
    )


    return params


def monogFertilityConfig(params: Parameters, N=30) -> MonogParameters:

    '''
    This function calculates the monogamous fertility
    function parameters.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    N: int
        resolution for the numerical integration
    '''
    p_k = float(params.k)
    c_k = gamma(p_k + 0.5) * (2 * p_k / np.pi) ** 0.5 / gamma(p_k + 1)
    cos_theta = np.cos(np.linspace(start=0, stop=2 * np.pi, num=N + 1)[:N])
    return MonogParameters(
        c_k=c_k,
        cosTheta=cos_theta
    )

def configure(params: Parameters) -> Parameters:

    '''
    This function defines a number of additional parameters.
    Parameters
    ----------
    params: dict
        dictionary containing the initial parameter names and values;
    Returns
    -------
    params: dict
        dictionary containing the updated parameter names and values;
    '''

    # level of discretization for the drawing of lifespans
    dT = 0.1

    # definition of the reproduction function
    if params.reproFuncName == 'epgMonog':
        params.monogParams = monogFertilityConfig(params)

    params.reproFunc = ParallelFuncs.mapper[params.reproFuncName]

    # max age cutoff point
    params.maxHostAge = np.min([np.max(params.muBreaks), np.max(params.contactAgeBreaks)])

    # full range of ages
    params.muAges = np.arange(start=0, stop=np.max(params.muBreaks), step=dT) + 0.5 * dT
    
    inner = np.digitize(params.muAges, params.muBreaks)-1
    params.hostMu = params.hostMuData[inner]

    # probability of surviving
    params.hostSurvivalCurve = np.exp(-np.cumsum(params.hostMu) * dT)

    # the index for the last age group before the cutoff in this discretization
    maxAgeIndex = np.argmax([params.muAges > params.maxHostAge]) - 1

    # cumulative probability of dying
    params.hostAgeCumulDistr = np.append(np.cumsum(dT * params.hostMu * np.append(1,
    params.hostSurvivalCurve[:-1]))[:maxAgeIndex], 1)

    params.contactAgeGroupBreaks = np.append(params.contactAgeBreaks[:-1], params.maxHostAge)
    params.treatmentAgeGroupBreaks = np.append(params.treatmentAgeBreaks[:-1], params.maxHostAge + dT)
    
    constructedVaccBreaks = np.sort(np.append(params.VaccTreatmentBreaks, params.VaccTreatmentBreaks + 1))
    a = np.append(-dT, constructedVaccBreaks)
    params.VaccTreatmentAgeGroupBreaks = np.append(a, params.maxHostAge + dT)

    if params.outTimings[-1] != params.maxTime:
        params.outTimings = np.append(params.outTimings, params.maxTime)



    return params




def setupSD(params: Parameters) -> SDEquilibrium:

    '''
    This function sets up the simulation to initial conditions
    based on analytical equilibria.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    SD: dict
        dictionary containing the equilibrium parameter settings;
    '''

    si = np.random.gamma(size=params.N, scale=1 / params.k, shape=params.k)
    sv = np.zeros(params.N, dtype=int)
    lifeSpans = getLifeSpans(params.N, params)
    trialBirthDates = - lifeSpans * np.random.uniform(low=0, high=1, size=params.N)
    trialDeathDates = trialBirthDates + lifeSpans
    sex_id = np.round(np.random.uniform(low = 1, high = 2, size = params.N))
    communityBurnIn = 1000

    while np.min(trialDeathDates) < communityBurnIn:

        earlyDeath = np.where(trialDeathDates < communityBurnIn)[0]
        trialBirthDates[earlyDeath] = trialDeathDates[earlyDeath]
        trialDeathDates[earlyDeath] += getLifeSpans(len(earlyDeath), params)

    demography = Demography(birthDate = trialBirthDates - communityBurnIn, deathDate = trialDeathDates - communityBurnIn)
    if params.contactAgeGroupBreaks is None:
        raise ValueError("contactAgeGroupBreaks not set")
    if params.treatmentAgeGroupBreaks is None:
        raise ValueError("treatmentAgeGroupBreaks not set")
    contactAgeGroupIndices = np.digitize(-demography.birthDate, params.contactAgeGroupBreaks)-1

    treatmentAgeGroupIndices = np.digitize(-demography.birthDate, params.treatmentAgeGroupBreaks)-1
    if params.equiData is None:
        raise ValueError("Equidata not set")
    
    meanBurdenIndex = np.digitize(-demography.birthDate, np.append(0, params.equiData.ageValues))-1

    wTotal = np.random.poisson(lam=si * params.equiData.stableProfile[meanBurdenIndex] * 2, size=params.N)
    worms = Worms(
        total = wTotal,
        female = np.random.binomial(n=wTotal, p=0.5, size=params.N)
    )
    stableFreeLiving = params.equiData.L_stable * 2
    if params.VaccTreatmentAgeGroupBreaks is None:
        raise ValueError("VaccTreatmentAgeGroupBreaks not set")
    VaccTreatmentAgeGroupIndices = np.digitize(-demography.birthDate, params.VaccTreatmentAgeGroupBreaks)-1


    SD = SDEquilibrium(
        si = si,
        sv = sv,
        worms = worms,
        sex_id = sex_id,
        freeLiving = stableFreeLiving,
        demography = demography,
        contactAgeGroupIndices = contactAgeGroupIndices,
        treatmentAgeGroupIndices = treatmentAgeGroupIndices,
        VaccTreatmentAgeGroupIndices = VaccTreatmentAgeGroupIndices,
        adherenceFactors = np.random.uniform(low=0, high=1, size=params.N),
        vaccinatedFactors = np.random.uniform(low=1, high=2, size=params.N),
        compliers = np.random.uniform(low=0, high=1, size=params.N) > params.propNeverCompliers,
        attendanceRecord = [],
        ageAtChemo = [],
        adherenceFactorAtChemo = [],
        vaccCount = 0,
        numSurvey = 0

    )

    return SD



def calcRates(params: Parameters, SD: SDEquilibrium):

    '''
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the equilibrium parameter values;
    Returns
    -------
    array of event rates;
    '''

    hostInfRates = SD.freeLiving * SD.si * params.contactRates[SD.contactAgeGroupIndices]
    deathRate = params.sigma * np.sum(SD.worms.total * params.v1[SD.sv])
    hostVaccDecayRates = params.VaccDecayRate[SD.sv]
    return np.append(hostInfRates, hostVaccDecayRates, deathRate)


def calcRates2(params: Parameters, SD: SDEquilibrium):

    '''
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.
    Each of these types of events happen to individual hosts.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the equilibrium parameter values;
    Returns
    -------
    array of event rates;
    '''
    hostInfRates = float(SD.freeLiving) * SD.si * params.contactRates[SD.contactAgeGroupIndices]
    deathRate = params.sigma * SD.worms.total * params.v1[SD.sv]
    hostVaccDecayRates = params.VaccDecayRate[SD.sv]
    args = (hostInfRates, hostVaccDecayRates, deathRate)
    return np.concatenate(args)

def doEvent(rates: NDArray[np.float_], params: Parameters, SD: SDEquilibrium) -> SDEquilibrium:

    '''
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries
    Parameters
    ----------
    rates: float
        array of event rates;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # determine which event takes place; if it's 1 to N, it's a new worm, otherwise it's a worm death
    event = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(rates) < np.cumsum(rates))

    if event == len(rates) - 1: # worm death event

        deathIndex = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(SD.worms.total * params.v1[SD.sv]) < np.cumsum(SD.worms.total* params.v1[SD.sv]))

        SD.worms.total[deathIndex] -= 1

        if np.random.uniform(low=0, high=1, size=1) < SD.worms.female[deathIndex] / SD.worms.total[deathIndex]:
            SD.worms.female[deathIndex] -= 1
    
    if event <= params.N:
        if np.random.uniform(low=0, high=1, size=1) < params.v3[SD.sv[event]]:
            SD.worms.total[event] += 1
            if np.random.uniform(low=0, high=1, size=1) < 0.5:
                SD.worms.female[event] += 1
    elif event <= 2*params.N:
        hostIndex = event - params.N
        SD.sv[hostIndex] = 0

    return SD



def doEvent2(rates: NDArray[np.float_], params: Parameters, SD: SDEquilibrium) -> SDEquilibrium:

    '''
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries
    Parameters
    ----------
    rates: float
        array of event rates;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    n_pop = params.N
    param_v3 = params.v3
    event = np.argmax(random.random() * np.sum(rates) < np.cumsum(rates))

    eventType = ((event) // n_pop) + 1
    hostIndex = ((event) % n_pop)
    
    if eventType == 1:
        if random.random() < param_v3[SD.sv[hostIndex]]:
            SD.worms.total[hostIndex] += 1
            if random.random() < 0.5:
                SD.worms.female[hostIndex] += 1
    elif eventType == 2:
        SD.sv[hostIndex] = 0
        
    elif eventType == 3:
        if random.random() < SD.worms.female[hostIndex]/SD.worms.total[hostIndex]:
            SD.worms.female[hostIndex] -= 1
        SD.worms.total[hostIndex] -= 1
    else:
        raise RuntimeError("Unknown event type")
        
    return SD
    
    
def doRegular(params: Parameters, SD: SDEquilibrium, t: int, dt: float) -> SDEquilibrium:
    '''
    This function runs processes that happen regularly.
    These processes are reincarnating whicever hosts have recently died and
    updating the free living worm population
    Parameters
    ----------
    rates: float
        array of event rates;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    
    t:  int
        time point;
    
    dt: float
        time interval;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    
    
    SD = doDeath(params, SD, t)
    SD = doFreeLive(params, SD, dt)
    return SD
    
def doFreeLive(params: Parameters, SD: SDEquilibrium, dt: float) -> SDEquilibrium:

    '''
    This function updates the freeliving population deterministically.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    dt: float
        time interval;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # polygamous reproduction; female worms produce fertilised eggs only if there's at least one male worm around
    if params.reproFuncName == 'epgFertility' and params.SR:
        productivefemaleworms = np.where(SD.worms.total == SD.worms.female, 0, SD.worms.female)

    elif params.reproFuncName == 'epgFertility' and not params.SR:
        productivefemaleworms = SD.worms.female

    # monogamous reproduction; only pairs of worms produce eggs
    elif params.reproFuncName == 'epgMonog':
        productivefemaleworms = np.minimum(SD.worms.total - SD.worms.female, SD.worms.female)

    eggOutputPerHost = params.lambda_egg * productivefemaleworms * np.exp(-SD.worms.total * params.gamma) * params.v2[SD.sv] # vaccine related fecundity
    eggsProdRate = 2 * params.psi * np.sum(eggOutputPerHost * params.rho[SD.contactAgeGroupIndices]) / params.N
    expFactor = np.exp(-params.LDecayRate * dt)
    SD.freeLiving = SD.freeLiving * expFactor + eggsProdRate * (1 - expFactor) / params.LDecayRate

    return SD

def doDeath(params: Parameters, SD: SDEquilibrium, t: int) -> SDEquilibrium:

    '''
    Death and aging function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # identify the indices of the dead
    theDead = np.where(SD.demography.deathDate < t)[0]

    if len(theDead) != 0:
        # they also need new force of infections (FOIs)
        SD.si[theDead] = np.random.gamma(size=len(theDead), scale=1 / params.k, shape=params.k)
        SD.sv[theDead] = 0
        #SD['sex_id'][theDead] = np.round(np.random.uniform(low = 1, high = 2, size = len(theDead)))
        # update the birth dates and death dates
        SD.demography.birthDate[theDead] = t - 0.001
        SD.demography.deathDate[theDead] = t + getLifeSpans(len(theDead), params)

        # kill all their worms
        SD.worms.total[theDead] = 0
        SD.worms.female[theDead] = 0

        # update the adherence factors
        SD.adherenceFactors[theDead] = np.random.uniform(low=0, high=1, size=len(theDead))

        # assign the newly-born to either comply or not
        SD.compliers[theDead] = np.random.uniform(low=0, high=1, size=len(theDead)) > params.propNeverCompliers
    assert params.contactAgeGroupBreaks is not None
    # update the contact age categories
    SD.contactAgeGroupIndices = np.digitize(t - SD.demography.birthDate, params.contactAgeGroupBreaks)-1
    
    assert params.treatmentAgeGroupBreaks is not None
    assert params.VaccTreatmentAgeGroupBreaks is not None
    # update the treatment age categories
    SD.treatmentAgeGroupIndices = np.digitize(t - SD.demography.birthDate, params.treatmentAgeGroupBreaks)-1
    SD.VaccTreatmentAgeGroupIndices = np.digitize(t - SD.demography.birthDate, params.VaccTreatmentAgeGroupBreaks)-1

    return SD

def doChemo(params: Parameters, SD: SDEquilibrium, t: NDArray[np.int_], coverage: ndarray) -> SDEquilibrium:

    '''
    Chemoterapy function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    coverage: array
        coverage fractions;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # decide which individuals are treated, treatment is random
    attendance = np.random.uniform(low=0, high=1, size=params.N) < coverage[SD.treatmentAgeGroupIndices]

    # they're compliers and it's their turn
    toTreatNow = np.logical_and(attendance, SD.compliers)

    # calculate the number of dead worms
    femaleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD.worms.female[toTreatNow],
    p=params.DrugEfficacy)

    maleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD.worms.total[toTreatNow] -
    SD.worms.female[toTreatNow], p=params.DrugEfficacy)

    SD.worms.female[toTreatNow] -= femaleToDie
    SD.worms.total[toTreatNow] -= (maleToDie + femaleToDie)

    # save actual attendance record and the age of each host when treated
    SD.attendanceRecord.append(toTreatNow)
    SD.ageAtChemo.append(t - SD.demography.birthDate)
    SD.adherenceFactorAtChemo.append(SD.adherenceFactors)

    return SD


def doChemoAgeRange(params: Parameters, SD: SDEquilibrium, t: int, minAge: int, maxAge: int, coverage: ndarray) -> SDEquilibrium:

    '''
    Chemoterapy function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    minAge: int
        minimum age for treatment;
    maxAge: int
        maximum age for treatment;
    coverage: array
        coverage fractions;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # decide which individuals are treated, treatment is random
    attendance = np.random.uniform(low=0, high=1, size=params.N) < coverage
    # get age of each individual
    ages = t - SD.demography.birthDate
    # choose individuals in correct age range
    correctAges = np.logical_and(ages <= maxAge , ages >= minAge)
    # they're compliers, in the right age group and it's their turn
    toTreatNow = np.logical_and(attendance, SD.compliers)
    toTreatNow = np.logical_and(toTreatNow , correctAges)
    
    # initialize the share of drug 1 and drug2 
    d1Share = 0
    d2Share = 0
    
    # get the actual share of each drug for this treatment.
    assert params.drug1Years is not None
    assert params.drug2Years is not None
    assert params.drug1Split is not None
    assert params.drug2Split is not None
    if t in params.drug1Years:
        i = np.where(params.drug1Years == t)[0][0]
        d1Share = params.drug1Split[i]
    if t in params.drug2Years:
        j = np.where(params.drug2Years == t)[0][0]
        d2Share = params.drug2Split[j]
        
    # assign which drug each person will take  
    drug = np.ones(int(sum(toTreatNow)))
    if d2Share > 0:
        k = random.sample(range(int(sum(drug))), int(sum(drug) * d2Share))
        drug[k] = 2
    # calculate the number of dead worms
    
    # if drug 1 share is > 0, then treat the appropriate individuals with drug 1
    if d1Share > 0:
        dEff = params.DrugEfficacy1
        k = np.where(drug == 1)[0]
        femaleToDie = np.random.binomial(size=len(k), n=np.array(SD.worms.female[k], dtype = 'int32'), p=dEff)
        maleToDie = np.random.binomial(size=len(k), n=np.array(SD.worms.total[k] - SD.worms.female[k], dtype = 'int32'), p=dEff)
        SD.worms.female[k] -= femaleToDie
        SD.worms.total[k] -= (maleToDie + femaleToDie)
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(k)
        SD.nChemo1 += len(k)
    # if drug 2 share is > 0, then treat the appropriate individuals with drug 2
    if d2Share > 0:
        dEff = params.DrugEfficacy2
        k = np.where(drug == 2)[0]
        femaleToDie = np.random.binomial(size=len(k), n=np.array(SD.worms.female[k], dtype = 'int32'), p=dEff)
        maleToDie = np.random.binomial(size=len(k), n=np.array(SD.worms.total[k] - SD.worms.female[k], dtype = 'int32'), p=dEff)
        SD.worms.female[k] -= femaleToDie
        SD.worms.total[k] -= (maleToDie + femaleToDie)
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(k)    
        SD.nChemo2 += len(k)
    
    SD.ageAtChemo.append(t - SD.demography.birthDate)
    SD.adherenceFactorAtChemo.append(SD.adherenceFactors)

    return SD



def doVaccine(params: Parameters, SD: SDEquilibrium, t: int, VaccCoverage: ndarray) -> SDEquilibrium:
    '''
    Vaccine function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    VaccCoverage: array
        coverage fractions;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    temp  = ((SD.VaccTreatmentAgeGroupIndices + 1 ) // 2) - 1
    vaccinate = np.random.uniform(low=0, high=1, size=params.N) < VaccCoverage[temp]
    
    indicesToVaccinate=[]
    for i in range(len(params.VaccTreatmentBreaks)):
        indicesToVaccinate.append(1+i*2)
    Hosts4Vaccination = []
    for i in SD.VaccTreatmentAgeGroupIndices:
        Hosts4Vaccination.append(i in indicesToVaccinate)
    
    vaccNow = np.logical_and(Hosts4Vaccination, vaccinate)
    SD.sv[vaccNow] = 1
    SD.vaccCount += sum(Hosts4Vaccination) + sum(vaccinate)
    
    return SD
    
    
    
    
    
def doVaccineAgeRange(params: Parameters, SD: SDEquilibrium, t: int, minAge: float, maxAge: float, coverage: ndarray) -> SDEquilibrium:
    '''
    Vaccine function.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    SD: dict
        dictionary containing the initial equilibrium parameter values;
    t: int
        time step;
    minAge: float
        minimum age for targeted vaccination;
    maxAge: float
        maximum age for targeted vaccination;    
    coverage: array
        coverage of vaccination ;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    vaccinate = np.random.uniform(low=0, high=1, size=params.N) < coverage
    ages = t - SD.demography.birthDate
    correctAges = np.logical_and(ages <= maxAge , ages >= minAge)
    # they're compliers and it's their turn
    vaccNow = np.logical_and(vaccinate, SD.compliers)
    vaccNow = np.logical_and(vaccNow , correctAges)
    SD.sv[vaccNow] = 1
    SD.vaccCount += sum(vaccNow) 
    
    return SD
    
    

def conductSurvey(SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int) -> Tuple[SDEquilibrium, float]:
    # get min and max age for survey
    minAge = params.minSurveyAge
    maxAge = params.maxSurveyAge
    #minAge = 5
    #maxAge = 15
    # get Kato-Katz eggs for each individual
    if nSamples < 1:
        raise ValueError('nSamples < 1')
    
    eggCounts = getSetOfEggCounts(SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized)
    for _ in range(nSamples -1):
        eggCounts = np.add(eggCounts, getSetOfEggCounts(SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized))
    
    eggCounts = eggCounts / nSamples

    # get individuals in chosen survey age group
    ages = -(SD.demography.birthDate - t)
    surveyAged = np.logical_and(ages >= minAge, ages <= maxAge)

    # get egg counts for those individuals
    surveyEggs = eggCounts[surveyAged]

    # get sampled individuals
    KKSampleSize = min(sampleSize, sum(surveyAged)) 

    sampledEggs = np.random.choice(a=np.array(surveyEggs), size=int(KKSampleSize), replace=False)
    SD.numSurvey += 1
    # return the prevalence
    return SD, np.sum(sampledEggs > 0.9) / KKSampleSize


def conductSurveyTwo(SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int) -> Tuple[SDEquilibrium, float]:

    # get Kato-Katz eggs for each individual
    if nSamples < 1:
        raise ValueError('nSamples < 1')
    eggCounts = getSetOfEggCounts(SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized)
    for _ in range(nSamples):
        eggCounts = np.add(eggCounts, getSetOfEggCounts(SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized))
    eggCounts = eggCounts / nSamples

    # get sampled individuals
    KKSampleSize = min(sampleSize, params.N) 
    sampledEggs = np.random.choice(a=eggCounts, size=KKSampleSize, replace=False)
    SD.numSurveyTwo += 1
    # return the prevalence
    return SD, np.sum(sampledEggs > 0.9) / KKSampleSize

    
    
def getPsi(params: Parameters) -> float:

    '''
    This function calculates the psi parameter.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    value of the psi parameter;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params.maxHostAge, step=deltaT) + 0.5 * deltaT




    inner = np.digitize(modelAges, params.muBreaks)-1

    # hostMu for the new age intervals
    hostMu = params.hostMuData[inner]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan: float = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT

    # calculate the cumulative sum of host and worm death rates from which to calculate worm survival
    # intMeanWormDeathEvents = np.cumsum(hostMu + params['sigma']) * deltaT # commented out as it is not used

    if params.contactAgeGroupBreaks is None:
        raise ValueError("contactAgeGroupBreaks is not set")
    modelAgeGroupCatIndex = np.digitize(modelAges, params.contactAgeGroupBreaks)-1

    betaAge = params.contactRates[modelAgeGroupCatIndex]
    rhoAge: float = params.rho[modelAgeGroupCatIndex]

    wSurvival = np.exp(-params.sigma * modelAges)

    B = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))])

    return params.R0 * MeanLifespan * params.LDecayRate / (params.lambda_egg * params.z * np.sum(rhoAge * hostSurvivalCurve * B) * deltaT)

def getLifeSpans(nSpans: int, params: Parameters) -> float:

    '''
    This function draws the lifespans from the population survival curve.
    Parameters
    ----------
    nSpans: int
        number of drawings;
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    array containing the lifespan drawings;
    '''
    if params.hostAgeCumulDistr is None:
        raise ValueError('hostAgeCumulDistr is not set')
    u = np.random.uniform(low=0, high=1, size=nSpans) * np.max(params.hostAgeCumulDistr)
    spans = np.array([np.argmax(u[i] < params.hostAgeCumulDistr) for i in range(nSpans)])
    if params.muAges is None:
        raise ValueError('muAges not set')
    else:
        return params.muAges[spans]



def getEquilibrium(params: Parameters) -> Equilibrium:

    '''
    This function returns a dictionary containing the equilibrium worm burden
    with age and the reservoir value as well as the breakpoint reservoir value
    and other parameter settings.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    Returns
    -------
    dictionary containing the equilibrium parameter settings;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params.maxHostAge, step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params.hostMuData[np.digitize(modelAges, params.muBreaks)-1]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT
    modelAgeGroupCatIndex = np.digitize(modelAges, params.contactAgeBreaks)-1

    betaAge = params.contactRates[modelAgeGroupCatIndex]
    rhoAge = params.rho[modelAgeGroupCatIndex]

    wSurvival = np.exp(-params.sigma * modelAges)

    # this variable times L is the equilibrium worm burden
    Q = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))])

    # converts L values into mean force of infection
    FOIMultiplier = np.sum(betaAge * hostSurvivalCurve) * deltaT / MeanLifespan

    # upper bound on L
    SRhoT = np.sum(hostSurvivalCurve * rhoAge) * deltaT
    R_power = 1 / (params.k + 1)
    L_hat = params.z * params.lambda_egg * params.psi * SRhoT * params.k * (params.R0 ** R_power - 1) / \
    (params.R0 * MeanLifespan * params.LDecayRate * (1 - params.z))

    # now evaluate the function K across a series of L values and find point near breakpoint;
    # L_minus is the value that gives an age-averaged worm burden of 1; negative growth should
    # exist somewhere below this
    L_minus = MeanLifespan / np.sum(Q * hostSurvivalCurve * deltaT)
    test_L: NDArray[np.float_] = np.append(np.linspace(start=0, stop=L_minus, num=10), np.linspace(start=L_minus, stop=L_hat, num=20))

    def K_valueFunc(currentL: float, params: Parameters) -> float:
        if params.reproFunc is None:
            raise ValueError('Reprofunc is not set')
        else:
            repro_result = params.reproFunc(currentL * Q, params)
        return params.psi * np.sum(repro_result * rhoAge * hostSurvivalCurve * deltaT) / (MeanLifespan * params.LDecayRate) - currentL
    #K_values = np.vectorize(K_valueFunc)(currentL=test_L, params=params)
    K_values = np.array([K_valueFunc(i,params) for i in test_L])
    
    # now find the maximum of K_values and use bisection to find critical Ls
    iMax = np.argmax(K_values)
    mid_L = test_L[iMax]

    if K_values[iMax] < 0:

        return Equilibrium(stableProfile=0 * Q,
                    ageValues=modelAges,
                    L_stable=0,
                    L_breakpoint=np.nan,
                    K_values=K_values,
                    L_values=test_L,
                    FOIMultiplier=FOIMultiplier)

    # find the top L
    L_stable = bisect(f=K_valueFunc, a=mid_L, b=4 * L_hat, args=(params))

    # find the unstable L
    L_break = test_L[1] / 50

    if K_valueFunc(L_break, params) < 0: # if it is less than zero at this point, find the zero
        L_break = bisect(f=K_valueFunc, a=L_break, b=mid_L, args=(params))

    stableProfile = L_stable * Q

    return Equilibrium(
        stableProfile=stableProfile, 
        ageValues=modelAges,
        hostSurvival=hostSurvivalCurve,
        L_stable=L_stable,
        L_breakpoint=L_break,
        K_values=K_values,
        L_values=test_L,
        FOIMultiplier=FOIMultiplier
    )

def extractHostData(results: List[List[Result]]):

    '''
    This function is used for processing results the raw simulation results.
    Parameters
    ----------
    results: list
        raw simulation output;
    Returns
    -------
    output: list
        processed simulation output;
    '''

    output = []

    for rep in range(len(results)):

        output.append(dict(
            wormsOverTime=np.array([results[rep][i].worms.total for i in range(len(results[0]) - 1)]).T,
            femaleWormsOverTime=np.array([results[rep][i].worms.female for i in range(len(results[0]) - 1)]).T,
            # freeLiving=np.array([results[rep][i]['freeLiving'] for i in range(len(results[0]) - 1)]),
            ages=np.array([results[rep][i].time - results[rep][i].hosts.birthDate for i in range(len(results[0]) - 1)]).T,
            # adherenceFactors=np.array([results[rep][i]['adherenceFactors'] for i in range(len(results[0]) - 1)]).T,
            # compliers=np.array([results[rep][i]['compliers'] for i in range(len(results[0]) - 1)]).T,
            # totalPop=len(results[rep][0]['worms']['total']),
            timePoints=np.array([np.array(results[rep][i].time) for i in range(len(results[0]) - 1)]),
            # attendanceRecord=results[rep][-1]['attendanceRecord'],
            # ageAtChemo=results[rep][-1]['ageAtChemo'],
            # finalFreeLiving=results[rep][-2]['freeLiving'],
            # adherenceFactorAtChemo=results[rep][-1]['adherenceFactorAtChemo']
            #sex_id = np.array([results[rep][i]['sex_id'] for i in range(len(results[0]) - 1)]).T
        ))

    return output

def getSetOfEggCounts(total: NDArray[np.int_], female: NDArray[np.int_], params: Parameters, Unfertilized: bool=False):

    '''
    This function returns a set of readings of egg counts from a vector of individuals,
    according to their reproductive biology.
    Parameters
    ----------
    total: int
        array of total worms;
    female: int
        array of female worms;
    params: dict
        dictionary containing the parameter names and values;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    Returns
    -------
    random set of egg count readings from a single sample;
    '''

    if Unfertilized:

        meanCount = female * params.lambda_egg * params.z ** female

    else:

        eggProducers = np.where(total == female, 0, female)
        meanCount = eggProducers * params.lambda_egg * params.z ** eggProducers

    return np.random.negative_binomial(size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg)

def getVillageMeanCountsByHost(villageList, timeIndex: int, params: Parameters, nSamples: int = 2, Unfertilized: bool = False):

    '''
    This function returns the mean egg count across readings by host
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    Returns
    -------
    array of mean egg counts;
    '''

    meanEggsByHost = getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
                                       villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    for i in range(1, nSamples):

        meanEggsByHost += getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
                                            villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    return meanEggsByHost

def getAgeCatSampledPrevByVillage(
    villageList, 
    timeIndex: int, 
    ageBand: int, 
    params: Parameters, 
    nSamples: int = 2, 
    Unfertilized: bool = False,
    villageSampleSize: int = 100) -> float:

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)


    ageGroups = np.digitize(villageList['ages'][:, timeIndex], np.append(-10, np.append(ageBand, 150)))-1

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(nSamples * mySample > 0.9) / villageSampleSize

def getAgeCatSampledPrevByVillageAll(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)
    ageGroups = np.digitize(villageList['ages'][:, timeIndex], np.append(-10, np.append(ageBand, 150)))-1

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]
    
    is_empty =currentAgeGroupMeanEggCounts.size == 0

    if(is_empty):
        infected =np.nan
        low = np.nan
        medium = np.nan
        heavy = np.nan
    else: 
        if villageSampleSize < len(currentAgeGroupMeanEggCounts):
            mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

        else:
            mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

        infected = np.sum(nSamples * mySample > 0.9) / villageSampleSize
        medium = np.sum((mySample >= params.mediumThreshold) & (mySample <= params.heavyThreshold)) / villageSampleSize
        heavy = np.sum(mySample > params.heavyThreshold) / villageSampleSize

        low = infected - (medium + heavy)

    return np.array(infected), np.array(low), np.array(medium), np.array(heavy), np.array(len(currentAgeGroupMeanEggCounts))


def getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)
    ageGroups = np.digitize(villageList['ages'][:, timeIndex], np.append(-10, np.append(ageBand, 150)))-1

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(mySample > 16) / villageSampleSize


def getSampledDetectedPrevByVillageAll(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    return [getAgeCatSampledPrevByVillageAll(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData]

def getBurdens(hostData, params, numReps, ageBand, nSamples=2, Unfertilized=False, villageSampleSize=100):
   
    results= np.empty((0,numReps))
    low_results = np.empty((0,numReps))
    medium_results = np.empty((0,numReps))
    heavy_results = np.empty((0,numReps))

    for t in range(len(hostData[0]['timePoints'])): #loop over time points
    # calculate burdens using the same sample
        newrow = np.array(getSampledDetectedPrevByVillageAll(hostData, t, ageBand, params, nSamples, Unfertilized, villageSampleSize))
        newrowinfected = newrow[:,0]
        newrowlow = newrow[:,1]
        newrowmedium = newrow[:,2]
        newrowheavy = newrow[:,3]
        # append row
        results = np.vstack([results, newrowinfected])
        low_results = np.vstack([low_results, newrowlow])
        medium_results = np.vstack([medium_results, newrowmedium])
        heavy_results = np.vstack([heavy_results, newrowheavy])

    # calculate proportion across number of repetitions
    prevalence = np.sum(results, axis = 1) / numReps
    low_prevalence = np.sum(low_results, axis = 1) / numReps
    medium_prevalence = np.sum(medium_results, axis = 1) / numReps
    heavy_prevalence = np.sum(heavy_results, axis = 1) / numReps

    return prevalence,low_prevalence,medium_prevalence,heavy_prevalence

def getSampledDetectedPrevByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getSampledDetectedPrevHeavyBurdenByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: dict
        dictionary containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getPrevalence(hostData, params, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100):

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    params: dict
        dictionary containing the parameter names and values;
    numReps: int
        number of simulations;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''

    sac_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([5, 15]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([16, 80]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([5, 15]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([16, 80]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_prevalence = np.sum(sac_results, axis=1) / numReps
    adult_prevalence = np.sum(adult_results, axis=1) / numReps

    sac_heavy_prevalence = np.sum(sac_heavy_results, axis=1) / numReps
    adult_heavy_prevalence = np.sum(adult_heavy_results, axis=1) / numReps

    df = pd.DataFrame({'Time': hostData[0]['timePoints'],
                       'SAC Prevalence': sac_prevalence,
                       'Adult Prevalence': adult_prevalence,
                       'SAC Heavy Intensity Prevalence': sac_heavy_prevalence,
                       'Adult Heavy Intensity Prevalence': adult_heavy_prevalence})

    df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    df['Time'] = df['Time'] - 50

    return df

def getPrevalenceDALYs(hostData, params, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100):

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    params: dict
        dictionary containing the parameter names and values;
    numReps: int
        number of simulations;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''


    #under 4s
    ufour_prevalence, ufour_low_prevalence, ufour_medium_prevalence, ufour_heavy_prevalence = getBurdens(hostData, params, numReps, np.array([0, 4]), nSamples=2, Unfertilized=False, villageSampleSize=100)

    # adults
    adult_prevalence, adult_low_prevalence, adult_medium_prevalence, adult_heavy_prevalence = getBurdens(hostData, params, numReps, np.array([5, 80]), nSamples=2, Unfertilized=False, villageSampleSize=100)

    #all individuals 
    all_prevalence, all_low_prevalence, all_medium_prevalence, all_heavy_prevalence = getBurdens(hostData, params, numReps, np.array([0, 80]), nSamples=2, Unfertilized=False, villageSampleSize=100)


    df = pd.DataFrame({'Time': hostData[0]['timePoints'],
                       'Prevalence': all_prevalence,
                        'Low Intensity Prevalence': all_low_prevalence,
                        'Medium Intensity Prevalence': all_medium_prevalence,
                        'Heavy Intensity Prevalence': all_heavy_prevalence,
                       'Under four Prevalence' : ufour_prevalence,
                        'Under four Low Intensity Prevalence': ufour_low_prevalence,
                       'Under four Medium Intensity Prevalence': ufour_medium_prevalence,
                       'Under four Heavy Intensity Prevalence': ufour_heavy_prevalence,
                        'Adult Prevalence': adult_prevalence,
                        'Adult Low Intensity Prevalence': adult_low_prevalence,
                        'Adult Medium Intensity Prevalence': adult_medium_prevalence,
                        'Adult Heavy Intensity Prevalence': adult_heavy_prevalence})


    df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    df['Time'] = df['Time'] - 50

    return df

def getPrevalenceDALYsAll(hostData, params: Parameters, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100) -> pd.DataFrame:

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: dict
        processed simulation output;
    params: dict
        dictionary containing the parameter names and values;
    numReps: int
        number of simulations;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''
    
    #all individuals 
    #all_prevalence, all_low_prevalence, all_medium_prevalence, all_heavy_prevalence = getBurdens(hostData, params, numReps, np.array([0, 80]), nSamples=2, Unfertilized=False, villageSampleSize=100)

    # df = pd.DataFrame({'Time': hostData[0]['timePoints'],
    #                    'Prevalence': all_prevalence,
    #                     'Low Intensity Prevalence': all_low_prevalence,
    #                     'Medium Intensity Prevalence': all_medium_prevalence,
    #                     'Heavy Intensity Prevalence': all_heavy_prevalence})
    
    for i in range(0,80) : #loop over yearly age bins
        
        prevalence, low_prevalence, moderate_prevalence, heavy_prevalence = getBurdens(hostData, params, numReps, np.array([i, i+1]), nSamples=2, Unfertilized=False, villageSampleSize=100)
        age_start = i
        age_end = i + 1
        #year = hostData[0]['timePoints']

        if i == 0:
            df = pd.DataFrame({
                   'Time': hostData[0]['timePoints'],
                   'age_start': np.repeat(age_start,len(low_prevalence)), 
                   'age_end':np.repeat(age_end,len(low_prevalence)), 
                   'intensity':np.repeat('light',len(low_prevalence)),
                   'species':np.repeat(params.species, len(low_prevalence)),
                   'measure':np.repeat('prevalence',len(low_prevalence)),
                   'draw_1':low_prevalence})
        
        else:
            df = df.append(pd.DataFrame({'Time':hostData[0]['timePoints'], 
                   'age_start': np.repeat(age_start,len(low_prevalence)), 
                   'age_end':np.repeat(age_end,len(low_prevalence)), 
                   'intensity':np.repeat('light',len(low_prevalence)),
                   'species':np.repeat(params.species,len(low_prevalence)),
                   'measure':np.repeat('prevalence',len(low_prevalence)),
                   'draw_1':low_prevalence}))
            
        df = df.append(pd.DataFrame({'Time':hostData[0]['timePoints'], 
                'age_start': np.repeat(age_start,len(low_prevalence)), 
                'age_end':np.repeat(age_end,len(low_prevalence)), 
                'intensity':np.repeat('moderate',len(low_prevalence)),
                'species':np.repeat(params.species,len(low_prevalence)),
                'measure':np.repeat('prevalence',len(low_prevalence)),
                'draw_1':moderate_prevalence}))
        
        df = df.append(pd.DataFrame({'Time':hostData[0]['timePoints'], 
                'age_start': np.repeat(age_start,len(low_prevalence)), 
                'age_end':np.repeat(age_end,len(low_prevalence)), 
                'intensity':np.repeat('heavy',len(low_prevalence)),
                'species':np.repeat(params.species,len(low_prevalence)),
                'measure':np.repeat('prevalence',len(low_prevalence)),
                'draw_1':heavy_prevalence}))
            
        # df[str(i)+' Prevalence'] = prevalence
        # df[str(i)+' Low Intensity Prevalence'] = low_prevalence
        # df[str(i)+' Medium Intensity Prevalence'] = medium_prevalence
        # df[str(i)+' Heavy Intensity Prevalence'] = heavy_prevalence



    #df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    #df['Time'] = df['Time'] - 50

    return df







def outputNumberInAgeGroup(results: List[List[Result]], params: Parameters):
    assert params.maxHostAge is not None
    for i in range(len(results[0])):
        d = results[0][i]
        ages = d.time - d.hosts.birthDate 
        ages1 = list(ages.astype(int))
        age_counts = []
        for j in range(int(params.maxHostAge)):
            age_counts.append(ages1.count(j))
            
        if i == 0:
                numEachAgeGroup = pd.DataFrame({
                        'Time': np.repeat(d.time, len(age_counts)), 
                       'age_start': range(int(params.maxHostAge)), 
                       'age_end':range(1,1+int(params.maxHostAge)), 
                       'intensity':np.repeat('All', len(age_counts)),
                       'species':np.repeat(params.species, len(age_counts)),
                       'measure':np.repeat('number', len(age_counts)),
                       'draw_1':age_counts})
        else:
                numEachAgeGroup = numEachAgeGroup.append(pd.DataFrame({
                        'Time': np.repeat(d.time, len(age_counts)), 
                       'age_start': range(int(params.maxHostAge)), 
                       'age_end':range(1,1+int(params.maxHostAge)), 
                       'intensity':np.repeat('All', len(age_counts)),
                       'species':np.repeat(params.species, len(age_counts)),
                       'measure':np.repeat('number', len(age_counts)),
                       'draw_1':age_counts}))
    
    return numEachAgeGroup 
