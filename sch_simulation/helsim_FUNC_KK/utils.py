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
    vaccState:NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    surveyType: str = 'KK2'
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

    
    # if Unfertilized:

    #     meanCount = female * params.lambda_egg * params.z**female * params.v2[vaccState]

    # else:
    if params.reproFuncName == "epgFertility" and params.SR:
            productivefemaleworms = np.where(
                total == female, 0, female
            )

    elif params.reproFuncName == "epgFertility" and not params.SR:
            productivefemaleworms = female

        # monogamous reproduction; only pairs of worms produce eggs
    elif params.reproFuncName == "epgMonog":
            productivefemaleworms = np.minimum(
                total - female, female
            )
        #eggProducers = np.where(total == female, 0, female)
        #meanCount = eggProducers * params.lambda_egg * params.z**eggProducers * params.v2[vaccState]
    meanCount = productivefemaleworms * params.lambda_egg * params.z**productivefemaleworms * params.v2[vaccState]
    eggs = np.random.negative_binomial(size=len(meanCount), 
                                           p=params.k_epg / (meanCount + params.k_epg), 
                                           n=params.k_epg)
        
    if surveyType == "KK2":
        eggs +=  np.random.negative_binomial(size=len(meanCount), 
                                             p=params.k_epg / (meanCount + params.k_epg), 
                                             n=params.k_epg)
    return eggs
        
        
def getSetOfEggCountsv2(
    total: NDArray[np.int_],
    female: NDArray[np.int_],
    vaccState:NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    surveyType: str = 'KK2'
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

    
    # if Unfertilized:

    #     meanCount = female * params.lambda_egg * params.z**female * params.v2[vaccState]

    # else:
    if params.reproFuncName == "epgFertility" and params.SR:
            productivefemaleworms = np.where(
                total == female, 0, female
            )

    elif params.reproFuncName == "epgFertility" and not params.SR:
            productivefemaleworms = female

        # monogamous reproduction; only pairs of worms produce eggs
    elif params.reproFuncName == "epgMonog":
            productivefemaleworms = np.minimum(
                total - female, female
            )
        #eggProducers = np.where(total == female, 0, female)
        #meanCount = eggProducers * params.lambda_egg * params.z**eggProducers * params.v2[vaccState]
    meanCount = productivefemaleworms * params.lambda_egg * params.z**productivefemaleworms * params.v2[vaccState]
        
    if surveyType == "KK1":
        # eggs = np.random.negative_binomial(size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg)
        # for i in range(nSamples):
        #     eggs +=  np.random.negative_binomial(
        #         size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg
        #         )
        mu_id = np.random.gamma(params.k_within, 
                                scale= meanCount/ params.k_within, 
                                size=len(female))
        mu_ids = np.random.gamma(params.k_slide * params.weight_sample / 0.025,
                                  scale= mu_id/params.k_slide, 
                                  size=len(female))
        count_ids = np.zeros(len(mu_ids))
        
        count_ids += np.random.poisson(mu_ids)
        return count_ids / params.weight_sample
        # return eggs
    if surveyType == "KK2":
        mu_id = np.random.gamma(params.k_within, 
                                scale= meanCount/ params.k_within, 
                                size=len(female))
        mu_ids = np.random.gamma(params.k_slide * params.weight_sample / 0.025,
                                  scale= mu_id/params.k_slide, 
                                  size=len(female))
        count_ids = np.zeros(len(mu_ids))
        for i in range(nSamples):
            count_ids += np.random.poisson(mu_ids)
            
        
        return count_ids / params.weight_sample
        # eggs = np.random.negative_binomial(size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg)
        # for i in range(nSamples):
        #     eggs +=  np.random.negative_binomial(
        #         size=len(meanCount), p=params.k_epg / (meanCount + params.k_epg), n=params.k_epg
        #         )
        # return eggs

def KKsampleGammaGammaPois(
    total: NDArray[np.int_],
    female: NDArray[np.int_],
    vaccState:NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool ,
    nSamples: int
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

    
    # if Unfertilized:

    #     meanCount = female * params.lambda_egg * params.z**female * params.v2[vaccState]

    # else:
    if params.reproFuncName == "epgFertility" and params.SR:
            productivefemaleworms = np.where(
                total == female, 0, female
            )

    elif params.reproFuncName == "epgFertility" and not params.SR:
            productivefemaleworms = female

        # monogamous reproduction; only pairs of worms produce eggs
    elif params.reproFuncName == "epgMonog":
            productivefemaleworms = np.minimum(
                total - female, female
            )
        #eggProducers = np.where(total == female, 0, female)
        #meanCount = eggProducers * params.lambda_egg * params.z**eggProducers * params.v2[vaccState]
    meanCount = productivefemaleworms * params.lambda_egg * params.z**productivefemaleworms * params.v2[vaccState]
        
        
    mu_id = np.random.gamma(params.k_within, 
                            scale= meanCount/ params.k_within, 
                            size=len(female))
    mu_ids = np.random.gamma(params.k_slide * params.weight_sample / 0.025,
                             scale= mu_id/params.k_slide, 
                             size=len(female))
    count_ids = 0
    for i in range(nSamples):
        count_ids += np.random.poisson(mu_ids)
    return count_ids / params.weight_sample
    
    
    
    
def POC_CCA_test(
    total: NDArray[np.int_],
    params: Parameters
) -> NDArray[np.int_]:

    """
    This function returns a set of readings of egg counts from a vector of individuals,
    according to their reproductive biology.
    Parameters
    ----------
    total: int
        array of total worms;
    params: Parameters object
        dataclass containing the parameter names and values;
    Returns
    -------
    positive of negative for each person;
    """

    antigen_test = 1*(np.random.uniform(size = len(total)) < (1 - np.power((1-params.testSensitivity), total)))
    x = np.where(total == 0)
    antigen_test[x[0]] = 1*np.random.uniform(size = len(x[0])) > params.testSpecificity
    
    return antigen_test
    


    
def PCR_test(
    total: NDArray[np.int_],
    female : NDArray[np.int_],
    params: Parameters
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
    Returns
    -------
    positive or negative for each person;
    """

    worm_pairs = np.minimum(
        total - female, female
    )
    
    antigen_test = 1*(np.random.uniform(size = len(worm_pairs)) < (1 - np.power((1-params.testSensitivity),worm_pairs)))
    x = np.where(worm_pairs == 0)
    antigen_test[x[0]] = 1*np.random.uniform(size = len(x[0])) > params.testSpecificity
    
    return antigen_test


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

def initializeTreatmentProbability(SD: SDEquilibrium, cov: float, snc: float) -> SDEquilibrium:

    if(snc > 0):
        alpha = cov * (1-snc)/snc
        beta = (1-cov)*(1-snc)/snc
        SD.treatProbability = np.random.beta(alpha, beta, len(SDEquilibrium.id))
    else:  # if snc = 0, then there is no variation between people in terms of probability of coverage
        SD.treatProbability = np.ones(len(SDEquilibrium.id)) * cov

    return SD


def checkForZeroTreatProbability(SD: SDEquilibrium, cov: float, snc: float) -> SDEquilibrium:
    # if there are any people with value -1 for treatment probability they need to get values for this 
    notInitTreatmentProb = np.where(SD.treatProbability == -1) 
    if len(notInitTreatmentProb) > 0:
        if(snc > 0):
            alpha = cov * (1-snc)/snc
            beta = (1-cov)*(1-snc)/snc
            SD.treatProbability[notInitTreatmentProb] = np.random.beta(alpha, beta, len(notInitTreatmentProb))
        else:
            SD.treatProbability[notInitTreatmentProb] = np.ones(len(notInitTreatmentProb)) * cov           
    return SD



def editTreatProbability(SD: SDEquilibrium, cov: float, snc: float) -> SDEquilibrium:
     
    if snc > 0:
        # Define the parameters for the probability of treatment
        alpha = cov * (1 - snc) / snc
        beta = (1 - cov) * (1 - snc) / snc

        # Draw probabilities from the beta distribution
        treatProbabilities = np.random.beta(alpha, beta, len(SDEquilibrium.id))
        # Sort these values so that they are in ascending order
        treatProbabilities.sort()

        # Store the current value of the treatProbability in an array
        oldTreatProbabilities = SD.treatProbability

        # Sort the indices array based on the values in oldTreatProbabilities
        indices = np.argsort(oldTreatProbabilities)
        # Assign the newly drawn treatment probabilities to the appropriate individuals
        for i, index in enumerate(indices):
            SD[index].treatProbability = treatProbabilities[i]
    else:
        SD.treatProbability = np.ones(len(SDEquilibrium.id)) * cov

    return SD