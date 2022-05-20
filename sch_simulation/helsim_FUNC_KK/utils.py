import warnings

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters, SDEquilibrium

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")


def getSetOfEggCounts(
    total: NDArray[np.int_],
    female: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool = False,
) -> NDArray[np.int_]:

    """
    This function returns a set of readings of egg counts from a vector of individuals,
    according to their reproductive biology.
    Parameters
    ----------
    total: int
        array of total worms;
    female: int
        array of female worms;
    params: Parameters object
        dataclass containing the parameter names and values;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    Returns
    -------
    random set of egg count readings from a single sample;
    """

    if Unfertilized:

        meanCount = female * params.lambda_egg * params.z**female

    else:

        eggProducers = np.where(total == female, 0, female)
        meanCount = eggProducers * params.lambda_egg * params.z**eggProducers

    return np.random.negative_binomial(
        size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg
    )


def calcRates(params: Parameters, SD: SDEquilibrium) -> ndarray:

    """
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the equilibrium parameter values;
    Returns
    -------
    array of event rates;
    """

    hostInfRates = (
        SD.freeLiving * SD.si * params.contactRates[SD.contactAgeGroupIndices]
    )
    deathRate = params.sigma * np.sum(SD.worms.total * params.v1[SD.sv])
    hostVaccDecayRates = params.VaccDecayRate[SD.sv]
    return np.append(hostInfRates, hostVaccDecayRates, deathRate)


def calcRates2(params: Parameters, SD: SDEquilibrium) -> NDArray[np.float_]:

    """
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.
    Each of these types of events happen to individual hosts.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the equilibrium parameter values;
    Returns
    -------
    array of event rates;
    """
    hostInfRates = (
        float(SD.freeLiving) * SD.si * params.contactRates[SD.contactAgeGroupIndices]
    )
    deathRate = params.sigma * SD.worms.total * params.v1[SD.sv]
    hostVaccDecayRates = params.VaccDecayRate[SD.sv]
    args = (hostInfRates, hostVaccDecayRates, deathRate)
    return np.concatenate(args)


def getLifeSpans(nSpans: int, params: Parameters) -> float:

    """
    This function draws the lifespans from the population survival curve.
    Parameters
    ----------
    nSpans: int
        number of drawings;
    params: Parameters object
        dataclass containing the parameter names and values;
    Returns
    -------
    array containing the lifespan drawings;
    """
    if params.hostAgeCumulDistr is None:
        raise ValueError("hostAgeCumulDistr is not set")
    u = np.random.uniform(low=0, high=1, size=nSpans) * np.max(params.hostAgeCumulDistr)
    # spans = np.array([np.argmax(v < params.hostAgeCumulDistr) for v in u])
    spans = np.argmax(
        params.hostAgeCumulDistr > u[:, None], axis=1
    )  # Should be faster?
    if params.muAges is None:
        raise ValueError("muAges not set")
    else:
        return params.muAges[spans]


def getPsi(params: Parameters) -> float:

    """
    This function calculates the psi parameter.
    Parameters
    ----------
    params: Parameters object
        dataclass containing the parameter names and values;
    Returns
    -------
    value of the psi parameter;
    """

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params.maxHostAge, step=deltaT) + 0.5 * deltaT

    inner = np.digitize(modelAges, params.muBreaks) - 1

    # hostMu for the new age intervals
    hostMu = params.hostMuData[inner]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan: float = np.sum(hostSurvivalCurve[: len(modelAges)]) * deltaT

    # calculate the cumulative sum of host and worm death rates from which to calculate worm survival
    # intMeanWormDeathEvents = np.cumsum(hostMu + params['sigma']) * deltaT # commented out as it is not used

    if params.contactAgeGroupBreaks is None:
        raise ValueError("contactAgeGroupBreaks is not set")
    modelAgeGroupCatIndex = np.digitize(modelAges, params.contactAgeGroupBreaks) - 1

    betaAge = params.contactRates[modelAgeGroupCatIndex]
    rhoAge: float = params.rho[modelAgeGroupCatIndex]

    wSurvival = np.exp(-params.sigma * modelAges)

    B = np.array(
        [
            np.sum(betaAge[:i] * np.flip(wSurvival[:i])) * deltaT
            for i in range(1, 1 + len(hostMu))
        ]
    )

    return (
        params.R0
        * MeanLifespan
        * params.LDecayRate
        / (
            params.lambda_egg
            * params.z
            * np.sum(rhoAge * hostSurvivalCurve * B)
            * deltaT
        )
    )
