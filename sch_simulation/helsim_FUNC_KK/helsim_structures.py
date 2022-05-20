import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")


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
    Years: ndarray  # 1-D array lower/upper
    Coverage: ndarray  # 1-D array


@dataclass
class Parameters:
    numReps: int
    maxTime: float  # nYears
    N: int  # nHosts
    R0: float  # Basic reproductive number
    lambda_egg: float  # Eggs per gram
    v2: NDArray[np.float_]  # Fraction of eggs produced when vaccinated.
    gamma: float  # Exponential density dependence of parasite adult stage
    k: float  # Shape parameter of assumed negative binomial distribution of worms amongst host
    sigma: float  # Worm death rate
    v1: NDArray[
        np.float_
    ]  # impact of vaccine on worm death rate KK. Assume worm death rate is v1*sigma.
    LDecayRate: float  # ReservoirDecayRate
    DrugEfficacy: float
    DrugEfficacy1: float
    DrugEfficacy2: float
    contactAgeBreaks: ndarray  # 1-D array - Contact age group breaks (minus sign necessary to include zero age)
    contactRates: ndarray  # 1-D array - BetaValues: Relative contact rates
    v3: ndarray  # 1-D array, v3 beta values: impact of vaccine on contact rates  Assume contact rate under vaccination is times v3. KK
    rho: ndarray  # 1-D array, - Rho, contribution to the reservoir by contact age group.
    treatmentAgeBreaks: ndarray  # 1-D array, treatmentBreaks Minimum age of each treatment group (minus sign necessary to include zero age): Infants; Pre-SAC; SAC; Adults
    VaccTreatmentBreaks: ndarray  # 1-D array, age range of vaccinated group.  ## KK: these are the lower bounds of ranges with width 1. THEY MUST BE > 1 YEAR APART!!
    coverage1: ndarray
    coverage2: ndarray
    VaccCoverage: ndarray  # Vaccine coverage of the age groups KK
    # VaccEfficacy
    treatInterval1: int  # interval between treatments in years.
    treatInterval2: int  # interval between treatments in years.
    treatStart1: float
    treatStart2: float
    nRounds1: int
    nRounds2: int
    chemoTimings1: ndarray  # 1-D array
    chemoTimings2: ndarray  # 1-D array
    VaccineTimings: ndarray  # 1-D array
    outTimings: ndarray  # 1-D array, outputEvents
    propNeverCompliers: float  # neverTreated
    highBurdenBreaks: ndarray  # 1-D array Three categories here
    highBurdenValues: ndarray  # 1-D array
    VaccDecayRate: ndarray  # vacc decay rate. rate of vaccine decay = 1/duration of vaccine   A vector with value 0 in state 1 and the vacc decay rate for state 2. KK.
    VaccTreatStart: float  ##Vaccine administration year start KK
    nRoundsVacc: int  ##number of vaccine rounds KK
    treatIntervalVacc: float  # KK
    heavyThreshold: int  # The threshold for heavy burden of infection, egg count > heavyThreshold
    mediumThreshold: int  # The threshold of medium burden of infection, mediumThreshold <= egg count <= heavyThreshold
    sampleSizeOne: int
    sampleSizeTwo: int
    nSamples: int
    minSurveyAge: float
    maxSurveyAge: float
    demogType: str  # demogName: subset of demography parameters to be extracted
    reproFuncName: str  # name of function for reproduction (a string).  [Deterministic] ## epgPerPerson   epgFertility	epgMonog
    z: float  # np.exp(-'gamma'),
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
    # Coverage
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
    numSurveyTwo: Optional[int] = None
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


@dataclass
class ProcResult:
    wormsOverTime: ndarray
    femaleWormsOverTime: ndarray
    ages: ndarray
    timePoints: ndarray
