import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import bisect
from scipy.special import gamma

import sch_simulation.helsim_FUNC_KK.ParallelFuncs as ParallelFuncs
from sch_simulation.helsim_FUNC_KK.helsim_structures import (
    Demography,
    Equilibrium,
    MonogParameters,
    Parameters,
    SDEquilibrium,
    Worms,
)
from sch_simulation.helsim_FUNC_KK.utils import getLifeSpans

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")


def monogFertilityConfig(params: Parameters, N: int = 30) -> MonogParameters:

    """
    This function calculates the monogamous fertility
    function parameters.

    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;

    N: int
        resolution for the numerical integration
    """
    p_k = float(params.k)
    c_k = gamma(p_k + 0.5) * (2 * p_k / np.pi) ** 0.5 / gamma(p_k + 1)
    cos_theta = np.cos(np.linspace(start=0, stop=2 * np.pi, num=N + 1)[:N])
    return MonogParameters(c_k=c_k, cosTheta=cos_theta)


def configure(params: Parameters) -> Parameters:

    """
    This function defines a number of additional parameters.
    Parameters
    ----------
    params: Parameters
        dataclass containing the initial parameter names and values;
    Returns
    -------
    params: Parameters
        dataclass containing the updated parameter names and values;
    """

    # level of discretization for the drawing of lifespans
    dT = 0.1

    # definition of the reproduction function
    if params.reproFuncName == "epgMonog":
        params.monogParams = monogFertilityConfig(params)

    params.reproFunc = ParallelFuncs.mapper[params.reproFuncName]

    # max age cutoff point
    params.maxHostAge = np.min(
        [np.max(params.muBreaks), np.max(params.contactAgeBreaks)]
    )

    # full range of ages
    params.muAges = np.arange(start=0, stop=np.max(params.muBreaks), step=dT) + 0.5 * dT

    inner = np.digitize(params.muAges, params.muBreaks) - 1
    params.hostMu = params.hostMuData[inner]

    # probability of surviving
    params.hostSurvivalCurve = np.exp(-np.cumsum(params.hostMu) * dT)

    # the index for the last age group before the cutoff in this discretization
    maxAgeIndex = np.argmax([params.muAges > params.maxHostAge]) - 1

    # cumulative probability of dying
    params.hostAgeCumulDistr = np.append(
        np.cumsum(dT * params.hostMu * np.append(1, params.hostSurvivalCurve[:-1]))[
            :maxAgeIndex
        ],
        1,
    )

    params.contactAgeGroupBreaks = np.append(
        params.contactAgeBreaks[:-1], params.maxHostAge
    )
    params.treatmentAgeGroupBreaks = np.append(
        params.treatmentAgeBreaks[:-1], params.maxHostAge + dT
    )

    constructedVaccBreaks = np.sort(
        np.append(params.VaccTreatmentBreaks, params.VaccTreatmentBreaks + 1)
    )
    a = np.append(-dT, constructedVaccBreaks)
    params.VaccTreatmentAgeGroupBreaks = np.append(a, params.maxHostAge + dT)

    if params.outTimings[-1] != params.maxTime:
        params.outTimings = np.append(params.outTimings, params.maxTime)

    return params


def setupSD(params: Parameters) -> SDEquilibrium:

    """
    This function sets up the simulation to initial conditions
    based on analytical equilibria.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the equilibrium parameter settings;
    """

    si = np.random.gamma(size=params.N, scale=1 / params.k, shape=params.k)
    sv = np.zeros(params.N, dtype=int)
    lifeSpans = getLifeSpans(params.N, params)
    trialBirthDates = -lifeSpans * np.random.uniform(low=0, high=1, size=params.N)
    trialDeathDates = trialBirthDates + lifeSpans
    sex_id = np.round(np.random.uniform(low=1, high=2, size=params.N))
    communityBurnIn = 1000

    while np.min(trialDeathDates) < communityBurnIn:

        earlyDeath = np.where(trialDeathDates < communityBurnIn)[0]
        trialBirthDates[earlyDeath] = trialDeathDates[earlyDeath]
        trialDeathDates[earlyDeath] += getLifeSpans(len(earlyDeath), params)

    demography = Demography(
        birthDate=trialBirthDates - communityBurnIn,
        deathDate=trialDeathDates - communityBurnIn,
    )
    if params.contactAgeGroupBreaks is None:
        raise ValueError("contactAgeGroupBreaks not set")
    if params.treatmentAgeGroupBreaks is None:
        raise ValueError("treatmentAgeGroupBreaks not set")
    contactAgeGroupIndices = (
        np.digitize(-demography.birthDate, params.contactAgeGroupBreaks) - 1
    )

    treatmentAgeGroupIndices = (
        np.digitize(-demography.birthDate, params.treatmentAgeGroupBreaks) - 1
    )
    if params.equiData is None:
        raise ValueError("Equidata not set")

    meanBurdenIndex = (
        np.digitize(-demography.birthDate, np.append(0, params.equiData.ageValues)) - 1
    )

    wTotal = np.random.poisson(
        lam=si * params.equiData.stableProfile[meanBurdenIndex] * 2, size=params.N
    )
    worms = Worms(
        total=wTotal, female=np.random.binomial(n=wTotal, p=0.5, size=params.N)
    )
    stableFreeLiving = params.equiData.L_stable * 2
    if params.VaccTreatmentAgeGroupBreaks is None:
        raise ValueError("VaccTreatmentAgeGroupBreaks not set")
    VaccTreatmentAgeGroupIndices = (
        np.digitize(-demography.birthDate, params.VaccTreatmentAgeGroupBreaks) - 1
    )

    SD = SDEquilibrium(
        si=si,
        sv=sv,
        worms=worms,
        sex_id=sex_id,
        freeLiving=stableFreeLiving,
        demography=demography,
        contactAgeGroupIndices=contactAgeGroupIndices,
        treatmentAgeGroupIndices=treatmentAgeGroupIndices,
        VaccTreatmentAgeGroupIndices=VaccTreatmentAgeGroupIndices,
        adherenceFactors=np.random.uniform(low=0, high=1, size=params.N),
        vaccinatedFactors=np.random.uniform(low=1, high=2, size=params.N),
        compliers=np.random.uniform(low=0, high=1, size=params.N)
        > params.propNeverCompliers,
        attendanceRecord=[],
        ageAtChemo=[],
        adherenceFactorAtChemo=[],
        vaccCount=0,
        numSurvey=0,
    )

    return SD


def getEquilibrium(params: Parameters) -> Equilibrium:

    """
    This function returns a dictionary containing the equilibrium worm burden
    with age and the reservoir value as well as the breakpoint reservoir value
    and other parameter settings.
    Parameters
    ----------
    params: Parameters object
        dataclass containing the parameter names and values;
    Returns
    -------
    dataclass containing the equilibrium parameter settings;
    """

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params.maxHostAge, step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params.hostMuData[np.digitize(modelAges, params.muBreaks) - 1]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[: len(modelAges)]) * deltaT
    modelAgeGroupCatIndex = np.digitize(modelAges, params.contactAgeBreaks) - 1

    betaAge = params.contactRates[modelAgeGroupCatIndex]
    rhoAge = params.rho[modelAgeGroupCatIndex]

    wSurvival = np.exp(-params.sigma * modelAges)

    # this variable times L is the equilibrium worm burden
    Q = np.array(
        [
            np.sum(betaAge[:i] * np.flip(wSurvival[:i])) * deltaT
            for i in range(1, 1 + len(hostMu))
        ]
    )

    # converts L values into mean force of infection
    FOIMultiplier = np.sum(betaAge * hostSurvivalCurve) * deltaT / MeanLifespan

    # upper bound on L
    SRhoT = np.sum(hostSurvivalCurve * rhoAge) * deltaT
    R_power = 1 / (params.k + 1)
    L_hat = (
        params.z
        * params.lambda_egg
        * params.psi
        * SRhoT
        * params.k
        * (params.R0**R_power - 1)
        / (params.R0 * MeanLifespan * params.LDecayRate * (1 - params.z))
    )

    # now evaluate the function K across a series of L values and find point near breakpoint;
    # L_minus is the value that gives an age-averaged worm burden of 1; negative growth should
    # exist somewhere below this
    L_minus = MeanLifespan / np.sum(Q * hostSurvivalCurve * deltaT)
    test_L: NDArray[np.float_] = np.append(
        np.linspace(start=0, stop=L_minus, num=10),
        np.linspace(start=L_minus, stop=L_hat, num=20),
    )

    def K_valueFunc(currentL: float, params: Parameters) -> float:
        if params.reproFunc is None:
            raise ValueError("Reprofunc is not set")
        else:
            repro_result = params.reproFunc(currentL * Q, params)
        return (
            params.psi
            * np.sum(repro_result * rhoAge * hostSurvivalCurve * deltaT)
            / (MeanLifespan * params.LDecayRate)
            - currentL
        )

    # K_values = np.vectorize(K_valueFunc)(currentL=test_L, params=params)
    K_values = np.array([K_valueFunc(i, params) for i in test_L])

    # now find the maximum of K_values and use bisection to find critical Ls
    iMax = np.argmax(K_values)
    mid_L = test_L[iMax]

    if K_values[iMax] < 0:

        return Equilibrium(
            stableProfile=0 * Q,
            ageValues=modelAges,
            L_stable=0,
            L_breakpoint=np.nan,
            K_values=K_values,
            L_values=test_L,
            FOIMultiplier=FOIMultiplier,
        )

    # find the top L
    L_stable = bisect(f=K_valueFunc, a=mid_L, b=4 * L_hat, args=(params))

    # find the unstable L
    L_break = test_L[1] / 50

    if (
        K_valueFunc(L_break, params) < 0
    ):  # if it is less than zero at this point, find the zero
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
        FOIMultiplier=FOIMultiplier,
    )
