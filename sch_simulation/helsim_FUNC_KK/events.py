import random
import warnings
from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters, SDEquilibrium
from sch_simulation.helsim_FUNC_KK.utils import getLifeSpans, getSetOfEggCounts, POC_CCA_test, PCR_test

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
        * np.exp(-productivefemaleworms * params.gamma)
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
        maxID = max(SD.id)
        new_ids = np.arange(maxID + 1, maxID + len(theDead) + 1)
        SD.id[theDead] = new_ids
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
    label:int,
    
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

    mda_t = t + label * 0.0001
    numChemo1 = 0
    numChemo2 = 0
    # decide which individuals are treated, treatment is random
    attendance = np.random.uniform(low=0, high=1, size=params.N) < coverage
    # get age of each individual
    ages = t - SD.demography.birthDate
    # choose individuals in correct age range
    correctAges = np.logical_and(ages < maxAge, ages >= minAge)
    # they're compliers, in the right age group and it's their turn
    toTreatNow = np.logical_and(attendance, SD.compliers)
    toTreatNow = np.logical_and(toTreatNow, correctAges)

    # initialize the share of drug 1 and drug2
    # we allow 2 different types of drug to be given within the same MDA.
    # Hence we need to split the population into groups who will take each of the drugs. 
    # in reality it will often be just one of the drugs each year, making this redundant
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

    # ensure that a drug is assigned even if missed in the coverage file    
    if np.logical_and(d1Share == 0, d2Share == 0):
        if max(params.drug1Years) < t:
            d2Share = 1
        else:
            d1Share = 1
    # assign which drug each person will take
    drug = np.ones(int(sum(toTreatNow)))

    if d2Share > 0:
        k = random.sample(range(int(sum(drug))), int(sum(drug) * d2Share))
        drug[k] = 2
# calculate the number of dead worms
    ll = np.where(toTreatNow==1)[0]
    # if drug 1 share is > 0, then treat the appropriate individuals with drug 1
    if d1Share > 0:
        dEff = params.DrugEfficacy1
        k = np.where(drug == 1)[0]

        femaleToDie = np.random.binomial(
            size=len(k), 
            n=np.array(SD.worms.female[ll[k]], dtype="int32"), 
            p=dEff
        )

        maleToDie = np.random.binomial(
            size=len(k),
            n=np.array(SD.worms.total[ll[k]] - SD.worms.female[ll[k]], dtype="int32"),
            p=dEff,
        )

        counts1, _ = np.histogram(ages[ll[k]], bins=np.arange(params.maxHostAge + 1))
        SD.worms.female[ll[k]] -= femaleToDie
        SD.worms.total[ll[k]] -= maleToDie + femaleToDie
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(ll[k])
        assert SD.nChemo1 is not None
        SD.nChemo1 += len(k)
        numChemo1 += len(k)
        SD.n_treatments[
            #str(mda_t) + ", MDA drug 1 (campaign " + str(label) + str(int(minAge)) + "-" + str(int(maxAge)) + ", )"
            str(mda_t) + ", MDA campaign " + str(label) + " (" + params.DrugName1 + ")"
        ] = counts1
        n_people_by_age, _ = np.histogram(
            ages,
            bins=np.arange(0, params.maxHostAge + 1),
        )
        SD.n_treatments_population[
            #str(mda_t) + ", MDA drug 1 (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(mda_t) + ", MDA campaign " + str(label) + " (" + params.DrugName1 + ")"
        ] = n_people_by_age
    # if drug 2 share is > 0, then treat the appropriate individuals with drug 2
    if d2Share > 0:
        dEff = params.DrugEfficacy2
        k = np.where(drug == 2)[0]

        femaleToDie = np.random.binomial(
            size=len(k),
            n=np.array(SD.worms.female[ll[k]], dtype="int32"),
            p=dEff
        )

        maleToDie = np.random.binomial(
            size=len(k),
            n=np.array(SD.worms.total[ll[k]] - SD.worms.female[ll[k]], dtype="int32"),
            p=dEff,
        )
        counts2, _ = np.histogram(ages[ll[k]], bins=np.arange(params.maxHostAge + 1))
        SD.worms.female[ll[k]] -= femaleToDie
        SD.worms.total[ll[k]] -= maleToDie + femaleToDie
        # save actual attendance record and the age of each host when treated
        SD.attendanceRecord.append(k)
        assert SD.nChemo2 is not None
        SD.nChemo2 += len(k)
        numChemo2 += len(k)
        SD.n_treatments[
            #str(mda_t) + ", MDA drug 2 (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(mda_t) + ", MDA campaign " + str(label) + " (" + params.DrugName2 + ")"
            
        ] = counts2

        n_people_by_age, _ = np.histogram(
            ages,
            bins=np.arange(0, params.maxHostAge + 1),
        )
        SD.n_treatments_population[
            #str(mda_t) + ", MDA drug 2 (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(mda_t) + ", MDA campaign " + str(label) + " (" + params.DrugName2 + ")"
        ] = n_people_by_age

    
    propTreated1 = numChemo1 / sum(correctAges)
    propTreated2 = numChemo2 / sum(correctAges)
    SD.ageAtChemo.append(t - SD.demography.birthDate)
    SD.adherenceFactorAtChemo.append(SD.adherenceFactors)

    return SD, propTreated1, propTreated2


def doVaccine(
    params: Parameters, 
    SD: SDEquilibrium, 
    t: int, 
    VaccCoverage: ndarray,
    label: int,
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
    vacc_t = t + label * 0.0001
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
    # get age of each individual
    ages = t - SD.demography.birthDate
    vaccs, _ = np.histogram(ages[vaccNow], bins=np.arange(params.maxHostAge + 1))
    SD.n_treatments[
            #str(vacc_t) + ", Vaccination (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(vacc_t) + ", Vaccination (campaign " + str(label) + ")"
        ] = vaccs
        
    n_people_by_age, _ = np.histogram(
            ages,
            bins=np.arange(0, params.maxHostAge + 1),
        )
    SD.n_treatments_population[
            #str(vacc_t) + ", Vaccination (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(vacc_t) + ", Vaccination (campaign " + str(label) + ")"
        ] = n_people_by_age
    return SD


def doVaccineAgeRange(
    params: Parameters,
    SD: SDEquilibrium,
    t: float,
    minAge: float,
    maxAge: float,
    coverage: ndarray,
    label: int,
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
    vacc_t = t + label * 0.0001
    vaccinate = np.random.uniform(low=0, high=1, size=params.N) < coverage
    ages = t - SD.demography.birthDate
    correctAges = np.logical_and(ages <= maxAge, ages >= minAge)
    # they're compliers and it's their turn
    #vaccNow = np.logical_and(vaccinate, SD.compliers)
    vaccNow = np.logical_and(vaccinate, correctAges)
    SD.sv[vaccNow] = 1
    SD.vaccCount += sum(vaccNow)
    propVacc = sum(vaccNow)/sum(correctAges)
    vaccs, _ = np.histogram(ages[vaccNow], bins=np.arange(params.maxHostAge + 1))
    SD.n_treatments[
            #str(vacc_t) + ", Vaccination (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(vacc_t) + ", Vaccination (campaign " + str(label) + ")"
        ] = vaccs
        
    n_people_by_age, _ = np.histogram(
            ages,
            bins=np.arange(0, params.maxHostAge + 1),
        )
    SD.n_treatments_population[
            #str(vacc_t) + ", Vaccination (" + str(int(minAge)) + "-" + str(int(maxAge)) + ")"
            str(vacc_t) + ", Vaccination (campaign " + str(label) + ")"
        ] = n_people_by_age
    return SD, propVacc


def doVectorControl(
    params: Parameters,
    SD: SDEquilibrium,
    vectorCoverage: ndarray,
    ) -> SDEquilibrium:
    """
    Vector control function.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    SD: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    vectorCoverage: array
        amount that freeliving larvae is reduced by;
    Returns
    -------
    SD: SDEquilibrium
        dataclass containing the updated equilibrium parameter values;
    """
    SD.freeLiving = SD.freeLiving * (1 - vectorCoverage)
    
    return SD

def conductKKSurvey(
        SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int, surveyType: str, writeSurvey: bool,
) -> Tuple[SDEquilibrium, float]:
    
    minAge = params.minSurveyAge
    maxAge = params.maxSurveyAge
   
    # get Kato-Katz eggs for each individual
    # if nSamples < 1:
    #     raise ValueError("nSamples < 1")



    if surveyType == 'KK1':
        nSamples = 1
    else: 
        nSamples = 2
        # eggCounts = KKsampleGammaGammaPois(
        #     SD.worms.total, SD.worms.female, SD.sv, params,  params.Unfertilized, nSamples
        #     )
    eggCounts = getSetOfEggCounts(SD.worms.total, SD.worms.female, SD.sv, params,  params.Unfertilized, nSamples, surveyType)
    
        
    eggCounts = eggCounts / nSamples

    # get individuals in chosen survey age group
    ages = -(SD.demography.birthDate - t)
    surveyAged = np.logical_and(ages >= minAge, ages <= maxAge)

    # get number of samples to take
    KKSampleSize = min(sampleSize, sum(surveyAged))
    # which people are in the required age range
    true_indices = np.where(surveyAged)[0]
    # sample from these people
    sample_indices = np.random.choice(true_indices, size=KKSampleSize, replace=False)
    # get the eggs for these sampled people
    sampledEggs = eggCounts[sample_indices]
    # get the number of people of each age that are surveyed
    surveys, _ = np.histogram(ages[sample_indices], bins=np.arange(params.maxHostAge + 1))
    if writeSurvey:
        # add this to the SD n surveys dict
        SD.n_surveys[
                str(t) + "," + str("surveys")
            ] = surveys
        # get the number of people in each age group
        n_people_by_age, _ = np.histogram(
                ages,
                bins=np.arange(0, params.maxHostAge + 1),
            )
        # add this to the SD n survey population dict
        SD.n_surveys_population[
                str(t) + "," + str("surveys")
            ] = n_people_by_age
        
    positivity = np.count_nonzero(sampledEggs) / KKSampleSize
    return SD, positivity 




def conductPOCCCASurvey(
        SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int,  writeSurvey: bool,
) -> Tuple[SDEquilibrium, float]:
    minAge = params.minSurveyAge
    maxAge = params.maxSurveyAge
    # minAge = 5
    # maxAge = 15
    # get Kato-Katz eggs for each individual
    
    
    POC_CCA_antigen = POC_CCA_test( SD.worms.total, params)
            
    
    # get individuals in chosen survey age group
    ages = -(SD.demography.birthDate - t)
    surveyAged = np.logical_and(ages >= minAge, ages <= maxAge)

    # get egg counts for those individuals
    surveyPOC_CCA = POC_CCA_antigen[surveyAged]

    # get sampled individuals
    POC_CCA_SampleSize = min(sampleSize, sum(surveyAged))
    # which people are in the required age range
    true_indices = np.where(surveyAged)[0]
    # sample from these people
    sample_indices = np.random.choice(true_indices, size=POC_CCA_SampleSize, replace=False)
    # get the eggs for these sampled people
    samples = surveyPOC_CCA[sample_indices]
    # get the number of people of each age that are surveyed
    surveys, _ = np.histogram(ages[sample_indices], bins=np.arange(params.maxHostAge + 1))
    if writeSurvey:
    # add this to the SD n surveys dict
        SD.n_surveys[
                str(t) + "," + str("surveys")
            ] = surveys
        # get the number of people in each age group
        n_people_by_age, _ = np.histogram(
                ages,
                bins=np.arange(0, params.maxHostAge + 1),
            )
        # add this to the SD n survey population dict
        SD.n_surveys_population[
                str(t) + "," + str("surveys")
            ] = n_people_by_age
        # sampledPOC_CCA = np.random.choice(
        #     a=np.array(surveyPOC_CCA), size=int(POC_CCA_SampleSize), replace=False
    # )
    positivity = sum(samples > 0) / POC_CCA_SampleSize
    return SD, positivity 


def conductPCRSurvey(
        SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int
) -> Tuple[SDEquilibrium, float]:
    minAge = params.minSurveyAge
    maxAge = params.maxSurveyAge
    # minAge = 5
    # maxAge = 15
    # get Kato-Katz eggs for each individual
    
    
    PCR_antigen = PCR_test( SD.worms.total, SD.worms.female, params)
            
    
    # get individuals in chosen survey age group
    ages = -(SD.demography.birthDate - t)
    surveyAged = np.logical_and(ages >= minAge, ages <= maxAge)

    # get egg counts for those individuals
    surveyPCR = PCR_antigen[surveyAged]

    # get sampled individuals
    PCR_SampleSize = min(sampleSize, sum(surveyAged))

    sampledPCR = np.random.choice(
        a=np.array(surveyPCR), size=int(PCR_SampleSize), replace=False
    )
    positivity = sum(sampledPCR > 0) / PCR_SampleSize
    return positivity 

def conductSurvey(
    SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int, surveyType: str, writeSurvey: bool
) -> Tuple[SDEquilibrium, float]:
    # get min and max age for survey
    if sampleSize > 0:
        if (surveyType == 'KK1') | (surveyType == 'KK2'):
            SD, positivity = conductKKSurvey(SD, params, t, sampleSize, nSamples, surveyType, writeSurvey)
        if surveyType == 'POC-CCA':
            SD, positivity = conductPOCCCASurvey(SD, params, t, sampleSize, writeSurvey)
        if surveyType == 'PCR':
            positivity = conductPCRSurvey(SD, params, t, sampleSize)
        SD.numSurvey += 1
    else:
        ages = -(SD.demography.birthDate - t)
        
        # get the number of people in each age group
        n_people_by_age, _ = np.histogram(
                ages,
                bins=np.arange(0, params.maxHostAge + 1),
            )
        # add this to the SD n survey population dict
        SD.n_surveys_population[
                str(t) + "," + str("surveys")
            ] = n_people_by_age
        SD.n_surveys[
                str(t) + "," + str("surveys")
            ] = np.zeros(len(n_people_by_age))
        positivity = 0
    # return the prevalence
    return SD, positivity


def conductSurveyTwo(
    SD: SDEquilibrium, params: Parameters, t: float, sampleSize: int, nSamples: int, surveyType: int
) -> Tuple[SDEquilibrium, float]:

    # get Kato-Katz eggs for each individual
    if nSamples < 1:
        raise ValueError("nSamples < 1")
    eggCounts = getSetOfEggCounts(
        SD.worms.total, SD.worms.female, SD.sv, params, params.Unfertilized,  nSamples,
        surveyType
    )
    for _ in range(nSamples):
        eggCounts = np.add(
            eggCounts,
            getSetOfEggCounts(
                SD.worms.total,
                SD.worms.female,
                SD.sv,
                params,
                params.Unfertilized,
                nSamples,
                surveyType
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
