import random
import warnings
from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters, SDEquilibrium
from sch_simulation.helsim_FUNC_KK.utils import getLifeSpans, getSetOfEggCounts

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")


def doEvent(
    rates: NDArray[np.float_], params: Parameters, SD: SDEquilibrium
) -> SDEquilibrium:

    """
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries
    Parameters
    ----------
    rates: float
        array of event rates;
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    # determine which event takes place; if it's 1 to N, it's a new worm, otherwise it's a worm death
    event = np.argmax(
        np.random.uniform(low=0, high=1, size=1) * np.sum(rates) < np.cumsum(rates)
    )

    if event == len(rates) - 1:  # worm death event

        deathIndex = np.argmax(
            np.random.uniform(low=0, high=1, size=1)
            * np.sum(SD.worms.total * params.v1[SD.sv])
            < np.cumsum(SD.worms.total * params.v1[SD.sv])
        )

        SD.worms.total[deathIndex] -= 1

        if (
            np.random.uniform(low=0, high=1, size=1)
            < SD.worms.female[deathIndex] / SD.worms.total[deathIndex]
        ):
            SD.worms.female[deathIndex] -= 1

    if event <= params.N:
        if np.random.uniform(low=0, high=1, size=1) < params.v3[SD.sv[event]]:
            SD.worms.total[event] += 1
            if np.random.uniform(low=0, high=1, size=1) < 0.5:
                SD.worms.female[event] += 1
    elif event <= 2 * params.N:
        hostIndex = event - params.N
        SD.sv[hostIndex] = 0

    return SD


def doEvent2(
    sum_rates: float,
    cumsum_rates: NDArray[np.float_],
    params: Parameters,
    SD: SDEquilibrium,
    multiplier: int = 1,
) -> SDEquilibrium:

    """
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries
    Parameters
    ----------
    sum_rates: float
        sum of array of event rates;
    cumsum_rates: NDArray[float]
        cumlative sum of event rates;
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """
    n_pop = params.N
    param_v3 = params.v3

    rand_array = np.random.uniform(size=multiplier) * sum_rates
    events_array = np.argmax(cumsum_rates > rand_array[:, None], axis=1)
    event_types_array = ((events_array) // n_pop) + 1
    host_index_array = (events_array) % n_pop

    event1_bools = event_types_array == 1
    event2_bools = event_types_array == 2
    event3_bools = event_types_array == 3
    event1_hosts = np.extract(event1_bools, host_index_array)
    event2_hosts = np.extract(event2_bools, host_index_array)
    event3_hosts = np.extract(event3_bools, host_index_array)

    param_v3s = np.take(param_v3, np.take(SD.sv, event1_hosts))
    event1_total_true_bools = np.random.uniform(size=len(event1_hosts)) < param_v3s
    event1_total_bools = np.full(len(event1_bools), False)
    np.place(event1_total_bools, event1_bools, event1_total_true_bools)

    total_array = np.where(event1_total_bools, 1, 0) + np.where(event3_bools, -1, 0)

    event3_worm_ratio = np.take(SD.worms.female, event3_hosts) / np.take(
        SD.worms.total, event3_hosts
    )
    event3_total_true_bools = (
        np.random.uniform(size=len(event3_hosts)) < event3_worm_ratio
    )
    event3_total_bools = np.full(len(event1_bools), False)
    np.place(event3_total_bools, event3_bools, event3_total_true_bools)
    females_array = np.where(
        np.logical_and(event1_total_bools, np.random.uniform(size=multiplier) < 0.5),
        1,
        0,
    ) + np.where(event3_total_bools, -1, 0)

    # Sort event 2
    np.put(SD.sv, event2_hosts, 0)
    # Sort event 1 & 3
    np.put(
        SD.worms.total,
        host_index_array,
        np.take(SD.worms.total, host_index_array) + total_array,
    )
    np.put(
        SD.worms.female,
        host_index_array,
        np.take(SD.worms.female, host_index_array) + females_array,
    )

    return SD


def doRegular(
    params: Parameters, SD: SDEquilibrium, t: int, dt: float
) -> SDEquilibrium:
    """
    This function runs processes that happen regularly.
    These processes are reincarnating whicever hosts have recently died and
    updating the free living worm population
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;

    t:  int
        time point;

    dt: float
        time interval;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    SD = doDeath(params, SD, t)
    SD = doFreeLive(params, SD, dt)
    return SD


def doFreeLive(params: Parameters, SD: SDEquilibrium, dt: float) -> SDEquilibrium:

    """
    This function updates the freeliving population deterministically.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    dt: float
        time interval;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    # polygamous reproduction; female worms produce fertilised eggs only if there's at least one male worm around
    if params.reproFuncName == "epgFertility" and params.SR:
        productivefemaleworms = np.where(
            SD.worms.total == SD.worms.female, 0, SD.worms.female
        )

    elif params.reproFuncName == "epgFertility" and not params.SR:
        productivefemaleworms = SD.worms.female

    # monogamous reproduction; only pairs of worms produce eggs
    elif params.reproFuncName == "epgMonog":
        productivefemaleworms = np.minimum(
            SD.worms.total - SD.worms.female, SD.worms.female
        )

    else:
        raise ValueError(f"Unsupported reproFuncName : {params.reproFuncName}")
    eggOutputPerHost = (
        params.lambda_egg
        * productivefemaleworms
        * np.exp(-SD.worms.total * params.gamma)
        * params.v2[SD.sv]
    )  # vaccine related fecundity
    eggsProdRate = (
        2
        * params.psi
        * np.sum(eggOutputPerHost * params.rho[SD.contactAgeGroupIndices])
        / params.N
    )
    expFactor = np.exp(-params.LDecayRate * dt)
    SD.freeLiving = (
        SD.freeLiving * expFactor + eggsProdRate * (1 - expFactor) / params.LDecayRate
    )

    return SD


def doDeath(params: Parameters, SD: SDEquilibrium, t: float) -> SDEquilibrium:

    """
    Death and aging function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    t: int
        time step;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    # identify the indices of the dead
    theDead = np.where(SD.demography.deathDate < t)[0]
    if len(theDead) != 0:
        # they also need new force of infections (FOIs)
        SD.si[theDead] = np.random.gamma(
            size=len(theDead), scale=1 / params.k, shape=params.k
        )
        SD.sv[theDead] = 0
        # SD['sex_id'][theDead] = np.round(np.random.uniform(low = 1, high = 2, size = len(theDead)))
        # update the birth dates and death dates
        SD.demography.birthDate[theDead] = t - 0.001
        SD.demography.deathDate[theDead] = t + getLifeSpans(len(theDead), params)

        # kill all their worms
        SD.worms.total[theDead] = 0
        SD.worms.female[theDead] = 0

        # update the adherence factors
        SD.adherenceFactors[theDead] = np.random.uniform(
            low=0, high=1, size=len(theDead)
        )

        # assign the newly-born to either comply or not
        SD.compliers[theDead] = (
            np.random.uniform(low=0, high=1, size=len(theDead))
            > params.propNeverCompliers
        )
    assert params.contactAgeGroupBreaks is not None
    # update the contact age categories
    SD.contactAgeGroupIndices = (
        np.digitize(t - SD.demography.birthDate, params.contactAgeGroupBreaks) - 1
    )

    assert params.treatmentAgeGroupBreaks is not None
    assert params.VaccTreatmentAgeGroupBreaks is not None
    # update the treatment age categories
    SD.treatmentAgeGroupIndices = (
        np.digitize(t - SD.demography.birthDate, params.treatmentAgeGroupBreaks) - 1
    )
    SD.VaccTreatmentAgeGroupIndices = (
        np.digitize(t - SD.demography.birthDate, params.VaccTreatmentAgeGroupBreaks) - 1
    )

    return SD


def doChemo(
    params: Parameters, SD: SDEquilibrium, t: NDArray[np.int_], coverage: ndarray
) -> SDEquilibrium:

    """
    Chemoterapy function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    t: int
        time step;
    coverage: array
        coverage fractions;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    # decide which individuals are treated, treatment is random
    attendance = (
        np.random.uniform(low=0, high=1, size=params.N)
        < coverage[SD.treatmentAgeGroupIndices]
    )

    # they're compliers and it's their turn
    toTreatNow = np.logical_and(attendance, SD.compliers)

    # calculate the number of dead worms
    femaleToDie = np.random.binomial(
        size=np.sum(toTreatNow), n=SD.worms.female[toTreatNow], p=params.DrugEfficacy
    )

    maleToDie = np.random.binomial(
        size=np.sum(toTreatNow),
        n=SD.worms.total[toTreatNow] - SD.worms.female[toTreatNow],
        p=params.DrugEfficacy,
    )

    SD.worms.female[toTreatNow] -= femaleToDie
    SD.worms.total[toTreatNow] -= maleToDie + femaleToDie

    # save actual attendance record and the age of each host when treated
    SD.attendanceRecord.append(toTreatNow)
    SD.ageAtChemo.append(t - SD.demography.birthDate)
    SD.adherenceFactorAtChemo.append(SD.adherenceFactors)

    return SD


def doChemoAgeRange(
    params: Parameters,
    SD: SDEquilibrium,
    t: float,
    minAge: int,
    maxAge: int,
    coverage: ndarray,
) -> SDEquilibrium:

    """
    Chemoterapy function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
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
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    # decide which individuals are treated, treatment is random
    attendance = np.random.uniform(low=0, high=1, size=params.N) < coverage
    # get age of each individual
    ages = t - SD.demography.birthDate
    # choose individuals in correct age range
    correctAges = np.logical_and(ages <= maxAge, ages >= minAge)
    # they're compliers, in the right age group and it's their turn
    toTreatNow = np.logical_and(attendance, SD.compliers)
    toTreatNow = np.logical_and(toTreatNow, correctAges)

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
        femaleToDie = np.random.binomial(
            size=len(k), n=np.array(SD.worms.female[k], dtype="int32"), p=dEff
        )
        maleToDie = np.random.binomial(
            size=len(k),
            n=np.array(SD.worms.total[k] - SD.worms.female[k], dtype="int32"),
            p=dEff,
        )
        SD.worms.female[k] -= femaleToDie
        SD.worms.total[k] -= maleToDie + femaleToDie
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(k)
        assert SD.nChemo1 is not None
        SD.nChemo1 += len(k)
    # if drug 2 share is > 0, then treat the appropriate individuals with drug 2
    if d2Share > 0:
        dEff = params.DrugEfficacy2
        k = np.where(drug == 2)[0]
        femaleToDie = np.random.binomial(
            size=len(k), n=np.array(SD.worms.female[k], dtype="int32"), p=dEff
        )
        maleToDie = np.random.binomial(
            size=len(k),
            n=np.array(SD.worms.total[k] - SD.worms.female[k], dtype="int32"),
            p=dEff,
        )
        SD.worms.female[k] -= femaleToDie
        SD.worms.total[k] -= maleToDie + femaleToDie
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(k)
        assert SD.nChemo2 is not None
        SD.nChemo2 += len(k)

    SD.ageAtChemo.append(t - SD.demography.birthDate)
    SD.adherenceFactorAtChemo.append(SD.adherenceFactors)

    return SD


def doVaccine(
    params: Parameters, SD: SDEquilibrium, t: int, VaccCoverage: ndarray
) -> SDEquilibrium:
    """
    Vaccine function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    t: int
        time step;
    VaccCoverage: array
        coverage fractions;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """
    assert SD.VaccTreatmentAgeGroupIndices is not None
    temp = ((SD.VaccTreatmentAgeGroupIndices + 1) // 2) - 1
    vaccinate = np.random.uniform(low=0, high=1, size=params.N) < VaccCoverage[temp]

    indicesToVaccinate = []
    for i in range(len(params.VaccTreatmentBreaks)):
        indicesToVaccinate.append(1 + i * 2)
    Hosts4Vaccination = []
    for i in SD.VaccTreatmentAgeGroupIndices:
        Hosts4Vaccination.append(i in indicesToVaccinate)

    vaccNow = np.logical_and(Hosts4Vaccination, vaccinate)
    SD.sv[vaccNow] = 1
    SD.vaccCount += sum(Hosts4Vaccination) + sum(vaccinate)

    return SD


def doVaccineAgeRange(
    params: Parameters,
    SD: SDEquilibrium,
    t: float,
    minAge: float,
    maxAge: float,
    coverage: ndarray,
) -> SDEquilibrium:
    """
    Vaccine function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    t: float
        time step;
    minAge: float
        minimum age for targeted vaccination;
    maxAge: float
        maximum age for targeted vaccination;
    coverage: array
        coverage of vaccination ;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """

    vaccinate = np.random.uniform(low=0, high=1, size=params.N) < coverage
    ages = t - SD.demography.birthDate
    correctAges = np.logical_and(ages <= maxAge, ages >= minAge)
    # they're compliers and it's their turn
    vaccNow = np.logical_and(vaccinate, SD.compliers)
    vaccNow = np.logical_and(vaccNow, correctAges)
    SD.sv[vaccNow] = 1
    SD.vaccCount += sum(vaccNow)

    return SD


def conductSurvey(
    SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int
) -> Tuple[SDEquilibrium, float]:
    # get min and max age for survey
    minAge = params.minSurveyAge
    maxAge = params.maxSurveyAge
    # minAge = 5
    # maxAge = 15
    # get Kato-Katz eggs for each individual
    if nSamples < 1:
        raise ValueError("nSamples < 1")

    eggCounts = getSetOfEggCounts(
        SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized
    )
    for _ in range(nSamples - 1):
        eggCounts = np.add(
            eggCounts,
            getSetOfEggCounts(
                SD.worms.total,
                SD.worms.female,
                params,
                Unfertilized=params.Unfertilized,
            ),
        )

    eggCounts = eggCounts / nSamples

    # get individuals in chosen survey age group
    ages = -(SD.demography.birthDate - t)
    surveyAged = np.logical_and(ages >= minAge, ages <= maxAge)

    # get egg counts for those individuals
    surveyEggs = eggCounts[surveyAged]

    # get sampled individuals
    KKSampleSize = min(sampleSize, sum(surveyAged))

    sampledEggs = np.random.choice(
        a=np.array(surveyEggs), size=int(KKSampleSize), replace=False
    )
    SD.numSurvey += 1
    # return the prevalence
    return SD, np.sum(sampledEggs > 0.9) / KKSampleSize


def conductSurveyTwo(
    SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int
) -> Tuple[SDEquilibrium, float]:

    # get Kato-Katz eggs for each individual
    if nSamples < 1:
        raise ValueError("nSamples < 1")
    eggCounts = getSetOfEggCounts(
        SD.worms.total, SD.worms.female, params, Unfertilized=params.Unfertilized
    )
    for _ in range(nSamples):
        eggCounts = np.add(
            eggCounts,
            getSetOfEggCounts(
                SD.worms.total,
                SD.worms.female,
                params,
                Unfertilized=params.Unfertilized,
            ),
        )
    eggCounts = eggCounts / nSamples

    # get sampled individuals
    KKSampleSize = min(sampleSize, params.N)
    sampledEggs = np.random.choice(a=eggCounts, size=KKSampleSize, replace=False)
    assert SD.numSurveyTwo is not None
    SD.numSurveyTwo += 1
    # return the prevalence
    return SD, np.sum(sampledEggs > 0.9) / KKSampleSize
