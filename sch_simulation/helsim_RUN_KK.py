import copy
import math
import multiprocessing
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from .helsim_FUNC_KK import (
    configuration,
    events,
    file_parsing,
    helsim_structures,
    results_processing,
    utils,
)

num_cores = multiprocessing.cpu_count()


def loadParameters(paramFileName: str, demogName: str) -> helsim_structures.Parameters:
    """
        This function loads all the parameters from the input text
    params    files and organizes them in a dictionary.
        helsim_structures.Parameters
        ----------
        paramFileName: str
            name of the input text file with the model parameters;
        demogName: str
            subset of demography parameters to be extracted;
        Returns
        -------
        params: dict
            dictionary containing the parameter names and values;
    """

    # load the parameters
    params = file_parsing.readParams(paramFileName=paramFileName, demogName=demogName)

    # configure the parameters
    params = configuration.configure(params)

    # update the parameters
    params.psi = utils.getPsi(params)
    params.equiData = configuration.getEquilibrium(params)

    return params


def doRealization(params, i, mult):
    """
    This function generates a single simulation path.
    helsim_structures.Parameters
    ----------
    params: helsim_structures.Parameters
        dataclass containing the parameter names and values;
    i: int
        iteration number;
    Returns
    -------
    results: List[helsim_structures.Result]
        list with simulation results;
    """
    # np.random.seed(i)
    # random.seed(i)
    params.equiData = configuration.getEquilibrium(params)
    # setup simulation data
    simData = configuration.setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params.maxTime)

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params.chemoTimings1)
    currentchemoTiming2 = copy.deepcopy(params.chemoTimings2)

    currentVaccineTimings = copy.deepcopy(params.VaccineTimings)

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min(
        np.array(
            [
                nextOutTime,
                t + maxStep,
                nextChemoTime1,
                nextChemoTime2,
                nextAgeTime,
                nextVaccineTime,
            ]
        )
    )

    results: List[
        helsim_structures.Result
    ] = []  # initialise empty list to store results
    multiplier = math.floor(
        params.N / 50
    )  # This appears to be the optimal value for all tests I've run - more or less than this takes longer!
    # run stochastic algorithm
    while t < maxTime:
        rates = utils.calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)
        new_multiplier = min(math.ceil(min((1 / 365) * sumRates, multiplier)), mult)
        # new_multiplier = math.floor(min((1/365) * sumRates, multiplier))
        # print(new_multiplier)
        new_multiplier = max(new_multiplier, 1)
        # new_multiplier = 5
        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:
            dt = 10000 * multiplier
        else:
            dt = np.random.exponential(scale=(1.0 / sumRates)) * new_multiplier
        if mult > 1:
            if t + dt >= nextStep:
                small_multiplier = np.array(list(range(new_multiplier, 0, -1)))
                dt1 = dt * small_multiplier / new_multiplier
                x = np.where((t + dt1) < nextStep)[0]
                if len(x) > 0:
                    x = x[0]
                    sm = small_multiplier[x]
                    dt = dt * sm / new_multiplier
                    new_multiplier = sm
                else:
                    sm = 1
                    dt = dt * sm / new_multiplier
                    new_multiplier = sm

        if t + dt < nextStep:
            t += dt
            simData = events.doEvent2(
                sumRates, cumsumRates, params, simData, new_multiplier
            )
            # simData = events.doFreeLive(params, simData, dt)
            # freeliveTime = t
        else:
            simData = events.doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:
                simData = events.doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime1:
                simData = events.doDeath(params, simData, t)
                simData = events.doChemo(params, simData, t, params.coverage1)

                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:
                simData = events.doDeath(params, simData, t)
                simData = events.doChemo(params, simData, t, params.coverage2)

                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:
                simData = events.doDeath(params, simData, t)
                simData = events.doVaccine(params, simData, t, params.VaccCoverage)
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]

            if timeBarrier >= nextOutTime:
                SD = copy.deepcopy(simData)
                eggCounts = utils.getSetOfEggCounts(
                    SD.worms.total,
                    SD.worms.female,
                    SD.sv,
                    params,
                    params.Unfertilized,
                    params.nSamples,
                    "KK2",
                )
                # get approximate prevalence of the population
                prev = len(np.where(eggCounts > 0)[0]) / len(eggCounts)
                results.append(
                    helsim_structures.Result(
                        iteration=1,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        si=copy.deepcopy(simData.si),
                        sv=copy.deepcopy(simData.sv),
                        contactAgeGroupIndices=copy.deepcopy(
                            simData.contactAgeGroupIndices
                        ),
                        nVacc=0,
                        nChemo1=0,
                        nChemo2=0,
                        nSurvey=0,
                        surveyPass=0,
                        elimination=0,
                        propChemo1=0,
                        propChemo2=0,
                        propVacc=0,
                        id=simData.id,
                        prevalence=prev,
                    )
                )
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min(
                [
                    nextOutTime,
                    t + maxStep,
                    nextChemoTime1,
                    nextChemoTime2,
                    nextVaccineTime,
                    nextAgeTime,
                ]
            )

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #    # ageAtChemo=np.array(simData['ageAtChemo']),
    #    # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results


def doRealizationSurveyCoveragePickle(
    params: helsim_structures.Parameters,
    surveyType: str,
    simData: Optional[helsim_structures.SDEquilibrium] = None,
    mult: int = 5,
) -> List[helsim_structures.Result]:
    """
    This function generates a single simulation path.
    helsim_structures.Parameters
    ----------
    params: helsim_structures.Parameters
        dataclass containing the parameter names and values;
    simData: helsim_structures.SDEquilibrium
        dataclass containing the initial equilibrium parameter values;
    i: int
        iteration number;
    Returns
    -------
    results: list
        list with simulation results;
    """
    if simData is None:
        simData = configuration.setupSD(params)

    # start time
    t: float = 0

    # end time
    maxTime = params.maxTime

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    (
        chemoTiming,
        VaccTiming,
        nextChemoTime,
        nextMDAAge,
        nextChemoIndex,
        nextVaccTime,
        nextVaccAge,
        nextVaccIndex,
        nextVecControlTime,
        nextVecControlIndex,
    ) = file_parsing.nextMDAVaccInfo(params)

    # next event

    nextStep = min(
        float(nextOutTime),
        float(t + maxStep),
        float(nextChemoTime),
        float(nextAgeTime),
        float(nextVaccTime),
        float(nextVecControlTime),
    )

    propChemo1 = []
    propChemo2 = []
    propVacc = []
    prevNChemo1 = 0
    prevNChemo2 = 0
    prevNVacc = 0
    nChemo = 0
    nVacc = 0
    nSurvey = 0
    surveyPass = 0
    tSurvey = maxTime + 10
    results = []  # initialise empty list to store results

    # run stochastic algorithm
    multiplier = math.floor(
        params.N / 50
    )  # This appears to be the optimal value for all tests I've run - more or less than this takes longer!
    while t < maxTime:
        rates = utils.calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)

        new_multiplier = min(math.ceil(min((1 / 365) * sumRates, multiplier)), mult)
        new_multiplier = max(new_multiplier, 1)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000

        if sumRates < 1e-4:
            dt = 10000 * multiplier

        else:
            dt = np.random.exponential(scale=(1.0 / sumRates)) * new_multiplier
        if mult > 1:
            if t + dt >= nextStep:
                small_multiplier = np.array(list(range(new_multiplier, 0, -1)))
                dt1 = dt * small_multiplier / new_multiplier
                x = np.where((t + dt1) < nextStep)[0]
                if len(x) > 0:
                    x = x[0]
                    sm = small_multiplier[x]
                    dt = dt * sm / new_multiplier
                    new_multiplier = sm
                else:
                    sm = 1
                    dt = dt * sm / new_multiplier
                    new_multiplier = sm

        new_t = t + dt
        if new_t < nextStep:
            t = new_t
            simData = events.doEvent2(
                sumRates, cumsumRates, params, simData, new_multiplier
            )
            # simData = events.doFreeLive(params, simData, dt)
            # freeliveTime = nextStep
        else:
            simData = events.doFreeLive(params, simData, nextStep - freeliveTime)
            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep
            # ageing and death
            if timeBarrier >= nextAgeTime:
                simData = events.doDeath(params, simData, t)

                nextAgeTime += ageingInt
            if timeBarrier >= nextOutTime:
                simData, _ = events.conductSurvey(
                    simData, params, t, params.N, params.nSamples, surveyType, False
                )
                SD = copy.deepcopy(simData)
                wormsTotal = sum(simData.worms.total)
                if wormsTotal == 0:
                    trueElim = 1
                else:
                    trueElim = 0
                eggCounts = utils.getSetOfEggCounts(
                    SD.worms.total,
                    SD.worms.female,
                    SD.sv,
                    params,
                    params.Unfertilized,
                    params.nSamples,
                    surveyType,
                )
                # get approximate prevalence of the population
                prev = len(np.where(eggCounts > 0)[0]) / len(eggCounts)
                # we have to check if this is the first time that the output is being done in order
                # to get the incidence in this time step, as if there are no previous results
                # then there is nothing to compare the currently infected people against
                if len(results) > 0:
                    # get ids of people who had 0 eggs last time step
                    previousZeros = results[-1].id[np.where(results[-1].eggCounts == 0)]
                    # get ids of people who have non-zero eggs this time step
                    nonZeros = SD.id[np.where(eggCounts > 0)]
                    # get the intersection of these ids, as this is the incidence in this timestep
                    pp = np.intersect1d(nonZeros, previousZeros)
                    # get the ages of these people to add to the results
                    incidenceAges = (
                        t - SD.demography.birthDate[np.where(np.isin(SD.id, pp))]
                    )
                else:
                    incidenceAges = []
                results.append(
                    helsim_structures.Result(
                        iteration=1,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        si=copy.deepcopy(simData.si),
                        sv=copy.deepcopy(simData.sv),
                        contactAgeGroupIndices=copy.deepcopy(
                            simData.contactAgeGroupIndices
                        ),
                        nVacc=simData.vaccCount - prevNVacc,
                        nChemo1=simData.nChemo1 - prevNChemo1,
                        nChemo2=simData.nChemo2 - prevNChemo2,
                        nSurvey=nSurvey,
                        surveyPass=surveyPass,
                        elimination=trueElim,
                        propChemo1=propChemo1,
                        propChemo2=propChemo2,
                        propVacc=propVacc,
                        prevalence=prev,
                        id=simData.id,
                        eggCounts=eggCounts,
                        incidenceAges=incidenceAges,
                    )
                )
                prevNChemo1 = simData.nChemo1
                prevNChemo2 = simData.nChemo2
                prevNVacc = simData.vaccCount
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]
            # survey
            if timeBarrier >= tSurvey:
                # print("Survey, time = ", t)
                simData, prevOne = events.conductSurvey(
                    simData,
                    params,
                    t,
                    params.sampleSizeOne,
                    params.nSamples,
                    surveyType,
                    True,
                )
                if params.sampleSizeOne > 0:
                    nSurvey += 1

                # if we pass the survey, then don't continue with MDA
                if prevOne < params.surveyThreshold:
                    # print("Passed survey, time = ", t)
                    surveyPass = 1
                    assert params.MDA is not None
                    for mda in params.MDA:
                        k = np.where(mda.Years > t + 1)
                        mda.Coverage[k] = np.array([0])
                    # assert params.Vacc is not None
                    # for vacc in params.Vacc:
                    #    vacc.Years = np.array([maxTime + 10])

                    # tSurvey = maxTime + 10
                    params.sampleSizeOne = 0
                # else:
                tSurvey = t + params.timeToNextSurvey

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = file_parsing.nextMDAVaccInfo(params)

            # chemotherapy
            if timeBarrier >= nextChemoTime:
                simData = events.doDeath(params, simData, t)
                assert params.MDA is not None
                for i in range(len(nextMDAAge)):
                    k = nextMDAAge[i]
                    index = nextChemoIndex[i]
                    cov = params.MDA[k].Coverage[index]
                    systematic_non_compliance = params.systematic_non_compliance
                    if (cov != simData.MDA_coverage) | (
                        systematic_non_compliance
                        != simData.MDA_systematic_non_compliance
                    ):
                        simData = utils.editTreatProbability(
                            simData, cov, systematic_non_compliance
                        )
                        simData.MDA_coverage = cov
                        simData.MDA_systematic_non_compliance = (
                            systematic_non_compliance
                        )
                    minAge = params.MDA[k].Age[0]
                    maxAge = params.MDA[k].Age[1]
                    label = params.MDA[k].Label
                    simData, propTreated1, propTreated2 = events.doChemoAgeRange(
                        params, simData, t, minAge, maxAge, cov, label
                    )
                    if propTreated1 > 0:
                        propChemo1.append(
                            [t, minAge, maxAge, "C1", round(propTreated1, 4)]
                        )
                    if propTreated2 > 0:
                        propChemo2.append(
                            [t, minAge, maxAge, "C2", round(propTreated2, 4)]
                        )

                if nChemo == 0:
                    tSurvey = t + params.timeToFirstSurvey
                if surveyType == "None":
                    tSurvey = maxTime + 10
                nChemo += 1

                params = file_parsing.overWritePostMDA(
                    params, nextMDAAge, nextChemoIndex
                )

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = file_parsing.nextMDAVaccInfo(params)

            # vaccination
            if timeBarrier >= nextVaccTime:
                #     break
                simData = events.doDeath(params, simData, t)
                assert params.Vacc is not None
                for i in range(len(nextVaccAge)):
                    k = nextVaccAge[i]
                    index = nextVaccIndex[i]
                    cov = params.Vacc[k].Coverage[index]
                    minAge = params.Vacc[k].Age[0]
                    maxAge = params.Vacc[k].Age[1]
                    label = params.Vacc[k].Label
                    simData, pVacc = events.doVaccineAgeRange(
                        params, simData, t, minAge, maxAge, cov, label
                    )
                    propVacc.append([t, minAge, maxAge, round(pVacc, 4)])

                nVacc += 1
                params = file_parsing.overWritePostVacc(
                    params, nextVaccAge, nextVaccIndex
                )

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = file_parsing.nextMDAVaccInfo(params)

            if timeBarrier >= nextVecControlTime:
                cov = params.VecControl[0].Coverage[nextVecControlIndex]
                simData = events.doVectorControl(params, simData, cov)
                params = file_parsing.overWritePostVecControl(
                    params, nextVecControlIndex
                )

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = file_parsing.nextMDAVaccInfo(params)

            nextStep = min(
                float(nextOutTime),
                float(t + maxStep),
                float(nextChemoTime),
                float(nextAgeTime),
                float(nextVaccTime),
                float(nextVecControlTime),
            )
    return results, simData


def singleSimulationDALYCoverage(
    params: helsim_structures.Parameters,
    simData: helsim_structures.SDEquilibrium,
    surveyType: str,
    numReps: Optional[int] = None,
) -> pd.DataFrame:
    """
    This function generates multiple simulation paths.
    helsim_structures.Parameters
    ----------
    params : helsim_structures.Parameters
        set of parameters for the run
    Returns
    -------
    df: data frame
        data frame with simulation results;
    """

    # extract the number of simulations
    if numReps is None:
        numReps = params.numReps

    # run the simulations
    results, SD = doRealizationSurveyCoveragePickle(params, surveyType, simData)

    resultslist = [results]
    # process the output
    output = results_processing.extractHostData(resultslist)

    # transform the output to data frame
    df = results_processing.getPrevalenceDALYsAll(
        output, params, numReps, params.Unfertilized, "KK1", 1
    )

    # wholePopPrev = results_processing.getPrevalenceWholePop(output, params, numReps, params.Unfertilized,  'KK1', 1)
    numAgeGroup = results_processing.outputNumberInAgeGroup(resultslist, params)
    incidence = results_processing.getIncidence(resultslist, params)
    costData = results_processing.getCostData(resultslist, params)
    allTimes = np.unique(numAgeGroup.Time)
    trueCoverageData = results_processing.getActualCoverages(
        resultslist, params, allTimes
    )
    surveyData = results_processing.outputNumberSurveyedAgeGroup(SD, params)
    treatmentData = results_processing.outputNumberTreatmentAgeGroup(SD, params)

    # df1 = pd.concat([wholePopPrev, df], ignore_index= True)
    df1 = pd.concat([df, numAgeGroup], ignore_index=True)
    df1 = pd.concat([df1, incidence], ignore_index=True)
    df1 = pd.concat([df1, costData], ignore_index=True)
    df1 = pd.concat([df1, trueCoverageData], ignore_index=True)
    df1 = pd.concat([df1, surveyData], ignore_index=True)
    df1 = pd.concat([df1, treatmentData], ignore_index=True)
    df1 = df1.reset_index()
    df1["draw_1"][np.where(pd.isna(df1["draw_1"]))[0]] = -1
    df1 = df1[
        ["Time", "age_start", "age_end", "intensity", "species", "measure", "draw_1"]
    ]
    return results, df1, SD


def getDesiredAgeDistribution(params, timeLimit):
    t = 0
    # initialize birth and death ages from population
    deathAges = getLifeSpans(params.N, params)
    birthAges = np.zeros(params.N)
    # age population for a chosen number of years in order to generate
    # wanted age distribution
    while t < timeLimit:
        t = min(deathAges)
        theDead = np.where(deathAges == t)[0]
        newDeathAges = getLifeSpans(len(theDead), params)
        deathAges[theDead] = newDeathAges + t
        birthAges[theDead] = t
    ages = t - birthAges
    deathAges = deathAges - t
    aa = np.zeros([len(ages), 2])
    aa[:, 0] = ages
    aa[:, 1] = deathAges
    b = aa
    ord1 = aa[:, 0].argsort()
    b[:, 0] = aa[ord1, 0]
    b[:, 1] = deathAges[ord1]
    return b


def splitSimDataIntoAges(ages, ageGroups):
    # this function should return an array which contains the location of individuals in the pickle
    # file who have age between 2 ages
    # ages is ages in pickle file
    # ageGroups is different splits in the ages to choose individuals from
    groupAges = []
    for i in range(1, len(ageGroups)):
        k1 = ages >= ageGroups[i - 1]
        k2 = ages < ageGroups[i]
        k = np.where(np.logical_and(k1, k2))

        groupAges.append(
            dict(minAge=ageGroups[i - 1], maxAge=ageGroups[i], indices=k[0])
        )

    return groupAges


def findNumberOfPeopleEachAgeGroup(chosenAges, groupAges):
    # how many people in each age group are in the wanted distribution
    # chosenAges is the age of people in the wanted distribution
    numIndivsToChoose = []
    for i in range(len(groupAges)):
        minAge = groupAges[i]["minAge"]
        maxAge = groupAges[i]["maxAge"]
        k1 = chosenAges >= minAge
        k2 = chosenAges < maxAge
        numIndivsToChoose.append(len(np.where(np.logical_and(k1, k2))[0]))
    return numIndivsToChoose


def selectIndividuals(chosenAges, groupAges, numIndivsToChoose):
    chosenIndivs = np.zeros(sum(numIndivsToChoose), dtype=int)
    startPoint = 0
    for i in range(len(numIndivsToChoose)):
        n1 = numIndivsToChoose[i]
        indivs = np.array(groupAges[i]["indices"])
        chosenIndivs[range(startPoint, startPoint + n1)] = np.random.choice(
            a=indivs, size=n1, replace=True
        )
        startPoint += n1
    return chosenIndivs


def multiple_simulations(
    params: helsim_structures.Parameters,
    pickleData,
    simparams,
    indices,
    i,
    surveyType,
    wantedPopSize=3000,
    ageGroups=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 100],
) -> pd.DataFrame:
    print(f"==> multiple_simulations starting sim {i}")
    start_time = time.time()
    # copy the parameters
    parameters = copy.deepcopy(params)
    j = indices[i]
    # load the previous simulation results
    data = pickleData[j]

    # extract the previous simulation output
    keys = [
        "si",
        "worms",
        "freeLiving",
        "demography",
        "contactAgeGroupIndices",
        "treatmentAgeGroupIndices",
    ]
    t = 0

    raw_data = dict((key, copy.deepcopy(data[key])) for key in keys)
    times = data["times"]
    raw_data["demography"]["birthDate"] = (
        raw_data["demography"]["birthDate"] - times["maxTime"]
    )
    raw_data["demography"]["deathDate"] = (
        raw_data["demography"]["deathDate"] - times["maxTime"]
    )
    worms = helsim_structures.Worms(
        total=raw_data["worms"]["total"], female=raw_data["worms"]["female"]
    )
    demography = helsim_structures.Demography(
        birthDate=raw_data["demography"]["birthDate"],
        deathDate=raw_data["demography"]["deathDate"],
    )
    ids = np.arange(len(raw_data["si"]))
    treatProbability = np.full(
        shape=len(raw_data["si"]), fill_value=np.NaN, dtype=float
    )
    simData = helsim_structures.SDEquilibrium(
        si=raw_data["si"],
        worms=worms,
        freeLiving=raw_data["freeLiving"],
        demography=demography,
        contactAgeGroupIndices=raw_data["contactAgeGroupIndices"],
        treatmentAgeGroupIndices=raw_data["treatmentAgeGroupIndices"],
        sv=np.zeros(len(raw_data["si"]), dtype=int),
        attendanceRecord=[],
        ageAtChemo=[],
        adherenceFactorAtChemo=[],
        n_treatments={},
        n_treatments_population={},
        n_surveys={},
        n_surveys_population={},
        vaccCount=0,
        nChemo1=0,
        nChemo2=0,
        numSurvey=0,
        compliers=np.random.uniform(low=0, high=1, size=len(raw_data["si"]))
        > params.propNeverCompliers,
        adherenceFactors=np.random.uniform(low=0, high=1, size=len(raw_data["si"])),
        id=ids,
        treatProbability=treatProbability,
    )
    pickleNumIndivs = len(simData.si)
    # print("starting  j =",j)
    if pickleNumIndivs < wantedPopSize:
        # set population size to wanted population size
        params.N = wantedPopSize
        # get the birth and death ages of representative population
        b = getDesiredAgeDistribution(params, timeLimit=200)
        chosenAges = b[:, 0]
        deathDate = b[:, 1]
        # ages of pickle file data
        ages = -simData.demography.birthDate
        # group these ages into age groups
        groupAges = splitSimDataIntoAges(ages, ageGroups)
        # how many people in each age group do we need to pick to match representative population
        numIndivsToChoose = findNumberOfPeopleEachAgeGroup(chosenAges, groupAges)
        # choose these people from the pickle data
        chosenIndivs = selectIndividuals(chosenAges, groupAges, numIndivsToChoose)
        birthDate = -chosenAges - 0.000001
        deathDate = deathDate
        wormsT = []
        wormsF = []
        si = []
        contactAgeGroupIndices = []
        treatmentAgeGroupIndices = []
        for k in range(len(chosenIndivs)):
            si.append(simData.si[chosenIndivs[k]])
            wormsT.append(simData.worms.total[chosenIndivs[k]])
            wormsF.append(simData.worms.female[chosenIndivs[k]])
            contactAgeGroupIndices.append(
                simData.contactAgeGroupIndices[chosenIndivs[k]]
            )
            treatmentAgeGroupIndices.append(
                simData.treatmentAgeGroupIndices[chosenIndivs[k]]
            )
        demography = helsim_structures.Demography(
            birthDate=np.array(birthDate),
            deathDate=np.array(deathDate),
        )
        worms = helsim_structures.Worms(total=np.array(wormsT), female=np.array(wormsF))
        ids = np.arange(wantedPopSize)
        SD = helsim_structures.SDEquilibrium(
            si=np.array(si),
            worms=worms,
            freeLiving=raw_data["freeLiving"],
            demography=demography,
            contactAgeGroupIndices=np.array(contactAgeGroupIndices),
            treatmentAgeGroupIndices=np.array(treatmentAgeGroupIndices),
            sv=np.zeros(wantedPopSize, dtype=int),
            attendanceRecord=[],
            ageAtChemo=[],
            adherenceFactorAtChemo=[],
            n_treatments={},
            n_treatments_population={},
            n_surveys={},
            n_surveys_population={},
            vaccCount=0,
            nChemo1=0,
            nChemo2=0,
            numSurvey=0,
            compliers=np.random.uniform(low=0, high=1, size=wantedPopSize)
            > params.propNeverCompliers,
            adherenceFactors=np.random.uniform(low=0, high=1, size=wantedPopSize),
            id=ids,
            treatProbability=np.full(
                shape=wantedPopSize, fill_value=np.NaN, dtype=float
            ),
        )
        simData = copy.deepcopy(SD)

    # Convert all layers to correct data format

    # extract the previous random state
    # state = data['state']
    # extract the previous simulation times

    simData.contactAgeGroupIndices = (
        np.digitize(
            np.array(t - simData.demography.birthDate),
            np.array(parameters.contactAgeGroupBreaks),
        )
        - 1
    )
    parameters.N = len(simData.si)

    # update the parameters
    R0 = simparams.iloc[j, 1].tolist()
    k = simparams.iloc[j, 2].tolist()
    parameters.R0 = R0
    parameters.k = k

    # configuration.configure the parameters
    parameters = configuration.configure(parameters)
    parameters.psi = utils.getPsi(parameters)
    parameters.equiData = configuration.getEquilibrium(parameters)
    # parameters['moderateIntensityCount'], parameters['highIntensityCount'] = setIntensityCount(paramFileName)

    # add a simulation path
    # results = doRealizationSurveyCoveragePickle(params, simData, 1)
    # output = results_processing.extractHostData(results)

    # transform the output to data frame
    df, simData = singleSimulationDALYCoverage(parameters, simData, surveyType, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> multiple_simulations finishing sim {i}: {total_time:.3f}s")
    return df, simData


def multiple_simulations_after_burnin(
    params: helsim_structures.Parameters,
    pickleData,
    simparams,
    indices,
    i,
    burnInTime,
    surveyType,
) -> pd.DataFrame:
    print(f"==> after burn in starting sim {i}")
    start_time = time.time()
    # copy the parameters
    parameters = copy.deepcopy(params)
    j = indices[i]
    # load the previous simulation results
    raw_data = pickleData[j]

    t = 0

    raw_data.demography.birthDate = raw_data.demography.birthDate - burnInTime
    raw_data.demography.deathDate = raw_data.demography.deathDate - burnInTime
    raw_data.n_treatments = {}
    raw_data.n_treatments_population = {}
    raw_data.n_surveys = {}
    raw_data.n_surveys_population = {}

    worms = helsim_structures.Worms(
        total=raw_data.worms.total, female=raw_data.worms.female
    )
    demography = helsim_structures.Demography(
        birthDate=raw_data.demography.birthDate,
        deathDate=raw_data.demography.deathDate,
    )
    simData = raw_data

    # Convert all layers to correct data format

    # extract the previous random state
    # state = data['state']
    # extract the previous simulation times

    simData.contactAgeGroupIndices = (
        np.digitize(
            np.array(t - simData.demography.birthDate),
            np.array(parameters.contactAgeGroupBreaks),
        )
        - 1
    )
    parameters.N = len(simData.si)

    # update the parameters
    R0 = simparams.iloc[j, 1].tolist()
    k = simparams.iloc[j, 2].tolist()
    parameters.R0 = R0
    parameters.k = k

    # configuration.configure the parameters
    parameters = configuration.configure(parameters)
    parameters.psi = utils.getPsi(parameters)
    parameters.equiData = configuration.getEquilibrium(parameters)
    # parameters['moderateIntensityCount'], parameters['highIntensityCount'] = setIntensityCount(paramFileName)

    # add a simulation path
    # results = doRealizationSurveyCoveragePickle(params, simData, 1)
    # output = results_processing.extractHostData(results)

    # transform the output to data frame
    results, df, simData = singleSimulationDALYCoverage(
        parameters, simData, surveyType, 1
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> after burn in finishing sim {i}: {total_time:.3f}s")
    return df, simData, results


def BurnInSimulations(
    params: helsim_structures.Parameters, simparams, i, surveyType
) -> pd.DataFrame:
    print(f"==> burn in starting sim {i}")
    start_time = time.time()
    # copy parameters
    parameters = copy.deepcopy(params)

    # update the parameters
    R0 = simparams.iloc[i, 1].tolist()
    k = simparams.iloc[i, 2].tolist()

    parameters.R0 = R0
    parameters.k = k
    # configuration.configure the parameters
    parameters = configuration.configure(parameters)
    parameters.psi = utils.getPsi(parameters)
    parameters.equiData = configuration.getEquilibrium(parameters)

    # setup starting point form simulation
    simData = configuration.setupSD(parameters)
    simData.vaccCount = 0
    simData.nChemo1 = 0
    simData.nChemo2 = 0
    simData.numSurvey = 0

    df, simData = singleSimulationDALYCoverage(parameters, simData, surveyType, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> burn in finishing sim {i}: {total_time:.3f}s")
    return df, simData


def getLifeSpans(nSpans: int, params: helsim_structures.Parameters) -> float:
    """
    This function draws the lifespans from the population survival curve.
    helsim_structures.Parameters
    ----------
    nSpans: int
        number of drawings;
    params: helsim_structures.Parameters object
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
