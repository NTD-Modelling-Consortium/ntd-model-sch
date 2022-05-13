import copy
import multiprocessing
import time

import cupy as np
from joblib import Parallel, delayed
import pandas as pd

from sch_simulation.helsim_FUNC_KK import (
    calcRates2,
    conductSurvey,
    configure,
    doChemo,
    doChemoAgeRange,
    doDeath,
    doEvent2,
    doFreeLive,
    doVaccine,
    getEquilibrium,
    getPsi,
    nextMDAVaccInfo,
    overWritePostMDA,
    readParams,
    setupSD,
    doVaccineAgeRange,
    overWritePostVacc,
    extractHostData,
    getPrevalence,
    getPrevalenceDALYsAll,
    outputNumberInAgeGroup,
    parse_coverage_input,
    readCoverageFile
)
num_cores = multiprocessing.cpu_count()


def loadParameters(paramFileName, demogName):
    '''
    This function loads all the parameters from the input text
params    files and organizes them in a dictionary.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    # load the parameters
    params = readParams(paramFileName=paramFileName, demogName=demogName)

    # configure the parameters
    params = configure(params)

    # update the parameters
    params['psi'] = getPsi(params)
    params['equiData'] = getEquilibrium(params)

    return params


def doRealization(params, i):
    '''
    This function generates a single simulation path.
    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;
    i: int
        iteration number;
    Returns
    -------
    results: list
        list with simulation results;
    '''

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params['chemoTimings1'])
    currentchemoTiming2 = copy.deepcopy(params['chemoTimings2'])

    currentVaccineTimings = copy.deepcopy(params['VaccineTimings'])

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                      nextChemoTime2, nextAgeTime, nextVaccineTime])

    results = []  # initialise empty list to store results

    # run stochastic algorithm
    while t < maxTime:
        if (t*1000 %10) == 0:
            print(t)
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:
            dt = 10000

        else:
            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:
            t += dt
            simData = doEvent2(rates, params, simData)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime1:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage1'])

                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage2'])

                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:

                simData = doDeath(params, simData, t)
                simData = doVaccine(params, simData, t, params['VaccCoverage'])
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]

            if timeBarrier >= nextOutTime:

                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    vaccState=copy.deepcopy(simData['sv']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    adherenceFactors=copy.deepcopy(
                        simData['adherenceFactors']),
                    compliers=copy.deepcopy(simData['compliers']),
                    sex_id = copy.deepcopy(simData['sex_id'])
                ))
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                              nextChemoTime2, nextVaccineTime, nextAgeTime])

    results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
        # ageAtChemo=np.array(simData['ageAtChemo']),
        # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    ))

    return results


def doRealizationSurvey(params, i):
    '''
    This function generates a single simulation path.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    i: int
        iteration number;

    Returns
    -------
    results: list
        list with simulation results;
    '''

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params['chemoTimings1'])
    currentchemoTiming2 = copy.deepcopy(params['chemoTimings2'])

    currentVaccineTimings = copy.deepcopy(params['VaccineTimings'])

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                      nextChemoTime2, nextAgeTime, nextVaccineTime])
    
    passSurveyOne = 0
   # passSurveyTwo = 0
    nChemo = 0
    nVacc = 0
    nSurvey = 0
   # nSurveyTwo = 0
    tSurvey = maxTime + 10
   # tSurveyTwo = maxTime + 10
    results = []  # initialise empty list to store results

    # run stochastic algorithm
    while t < maxTime:

        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:


            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt
   

            simData = doEvent2(rates, params, simData)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime1:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage1'])
                if nChemo == 0:
                    tSurvey = t + 5
                  #  tSurveyTwo = t + 9
                nChemo += 1
                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage2'])
                if nChemo == 0:
                    tSurvey = t + 5
                nChemo += 1
                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:

                simData = doDeath(params, simData, t)
                simData = doVaccine(params, simData, t, params['VaccCoverage'])
                nVacc += 1
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
            
            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(simData, params, t, params['sampleSizeOne'], nSamples)
                nSurvey += 1
                #tSurvey = maxTime + 10
                if prevOne < 0.01:
                    nextChemoTime1 = maxTime + 10
                    nextChemoTime2 = maxTime + 10
                    nextVaccineTime = maxTime + 10
                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + 4
                   # simData, prevTwo = conductSurveyTwo(simData, params, t, sampleSizeTwo, nSamples)
                   # nSurveyTwo += 1
                   # if prevTwo < 0.01:
                   #     nextChemoTime1 = maxTime + 10
                   #     nextChemoTime2 = maxTime + 10
                   #     nextVaccineTime = maxTime + 10
                     #   tSurveyTwo = maxTime + 10

            # if timeBarrier >= tSurveyTwo:
            #     simData, prevTwo = conductSurveyTwo(simData, params, t, sampleSizeTwo, nSamples)
            #     nSurveyTwo += 1
            #     if prevTwo < 0.01:
            #         nextChemoTime1 = maxTime + 10
            #         nextChemoTime2 = maxTime + 10
            #         nextVaccineTime = maxTime + 10
            #         tSurveyTwo = maxTime + 10
            #     else:
            #         tSurveyTwo = t + 4
                    
            if timeBarrier >= nextOutTime:

                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    vaccState=copy.deepcopy(simData['sv']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    adherenceFactors=copy.deepcopy(
                        simData['adherenceFactors']),
                    compliers=copy.deepcopy(simData['compliers']),
                    nVacc = nVacc,
                    nChemo = nChemo,
                    nSurvey = nSurvey #,
                  #  nSurveyTwo = nSurveyTwo
                ))
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                              nextChemoTime2, nextVaccineTime, nextAgeTime])

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #     # ageAtChemo=np.array(simData['ageAtChemo']),
    #     # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results




def doRealizationSurveyCoverage(params, i):
    '''
    This function generates a single simulation path.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    i: int
        iteration number;

    Returns
    -------
    results: list
        list with simulation results;
    '''

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52
    
    chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
    
    
    # next event
    
    
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime,
                       nextAgeTime, nextVaccTime])
    

    nChemo = 0
    nVacc = 0
    nSurvey = 0
  
    tSurvey = maxTime + 10
    results = []  # initialise empty list to store results
    print_t_interval = 0.5
    print_t = 0
    # run stochastic algorithm
    while t < maxTime:
        if t > print_t:
            print(t)
            print_t += print_t_interval
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:


            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt
   

            simData = doEvent2(rates, params, simData)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime:
                
                simData = doDeath(params, simData, t)
                for i in range(len(nextMDAAge)):
                    k = nextMDAAge[i]
                    index = nextChemoIndex[i]
                    cov = params['MDA_Coverage' + str(k)][index]
                    minAge = params['MDA_age'+str(k)][0]
                    maxAge = params['MDA_age'+str(k)][1]
                    simData = doChemoAgeRange(params, simData, t, minAge, maxAge, cov)
                if nChemo == 0:
                    tSurvey = t + params['timeToFirstSurvey']
                nChemo += 1
                
                params = overWritePostMDA(params,  nextMDAAge, nextChemoIndex)
                
                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
                

            # vaccination
            if timeBarrier >= nextVaccTime:

                simData = doDeath(params, simData, t)
                for i in range(len(nextVaccAge)):
                    k = nextVaccAge[i]
                    index = nextVaccIndex[i]
                    cov = params['Vacc_Coverage' + str(k)][index]
                    minAge = params['Vacc_age'+str(k)][0]
                    maxAge = params['Vacc_age'+str(k)][1]
                    simData = doVaccineAgeRange(params, simData, t, minAge, maxAge, cov)
               
                nVacc += 1
                params = overWritePostVacc(params,  nextVaccAge, nextVaccIndex)

                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
                
            # survey
            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(simData, params, t, params['sampleSizeOne'], params['nSamples'])
                nSurvey += 1

                if prevOne < params['surveyThreshold']:
                    for i in range(params['nMDAAges']):
                        k = i + 1
                        params['MDA_Years' + str(k)] = [maxTime + 10]
                    for i in range(params['nVaccAges']):
                        k = i + 1
                        params['Vacc_Years' + str(k)] = [maxTime + 10]
                        
                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + params['timeToNextSurvey']
                 
                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params) 
            
            if timeBarrier >= nextOutTime:

                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    vaccState=copy.deepcopy(simData['sv']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    adherenceFactors=copy.deepcopy(
                        simData['adherenceFactors']),
                    compliers=copy.deepcopy(simData['compliers']),
                    nVacc = nVacc,
                    nChemo = nChemo,
                    nSurvey = nSurvey
                ))
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime,
                       nextAgeTime, nextVaccTime])

    results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
        # ageAtChemo=np.array(simData['ageAtChemo']),
        # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    ))

    return results




def doRealizationSurveyCoveragePickle(params, simData, i):
    '''
    This function generates a single simulation path.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    i: int
        iteration number;

    Returns
    -------
    results: list
        list with simulation results;
    '''

   
    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52
    
    chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
    
    
    # next event
    
    
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime,
                       nextAgeTime, nextVaccTime])
    

    nChemo = 0
    nVacc = 0
    nSurvey = 0
    surveyPass = 0
    tSurvey = maxTime + 10
    results = []  # initialise empty list to store results
    print_t_interval = 0.5
    print_t = 0
    # run stochastic algorithm
    while t < maxTime:
        if t > print_t:
            print(t)
            print_t += print_t_interval
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:


            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt
   

            simData = doEvent2(rates, params, simData)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime:
                
                simData = doDeath(params, simData, t)
                for i in range(len(nextMDAAge)):
                    k = nextMDAAge[i]
                    index = nextChemoIndex[i]
                    cov = params['MDA_Coverage' + str(k)][index]
                    minAge = params['MDA_age'+str(k)][0]
                    maxAge = params['MDA_age'+str(k)][1]
                    simData = doChemoAgeRange(params, simData, t, minAge, maxAge, cov)
                if nChemo == 0:
                    tSurvey = t + params['timeToFirstSurvey']
                nChemo += 1
                
                params = overWritePostMDA(params,  nextMDAAge, nextChemoIndex)
                
                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
                

            # vaccination
            if timeBarrier >= nextVaccTime:

                simData = doDeath(params, simData, t)
                for i in range(len(nextVaccAge)):
                    k = nextVaccAge[i]
                    index = nextVaccIndex[i]
                    cov = params['Vacc_Coverage' + str(k)][index]
                    minAge = params['Vacc_age'+str(k)][0]
                    maxAge = params['Vacc_age'+str(k)][1]
                    simData = doVaccineAgeRange(params, simData, t, minAge, maxAge, cov)
               
                nVacc += 1
                params = overWritePostVacc(params,  nextVaccAge, nextVaccIndex)

                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params)
                
            # survey
            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(simData, params, t, params['sampleSizeOne'], params['nSamples'])
                nSurvey += 1

                if prevOne < params['surveyThreshold']:
                    surveyPass = 1
                    for i in range(params['nMDAAges']):
                        k = i + 1
                        params['MDA_Years' + str(k)] = [maxTime + 10]
                    for i in range(params['nVaccAges']):
                        k = i + 1
                        params['Vacc_Years' + str(k)] = [maxTime + 10]
                        
                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + params['timeToNextSurvey']
                 
                chemoTiming, VaccTiming, nextChemoTime, nextMDAAge, nextChemoIndex, nextVaccTime, nextVaccAge, nextVaccIndex = nextMDAVaccInfo(params) 
            
                    
            if timeBarrier >= nextOutTime:
                a, truePrev = conductSurvey(simData, params, t, params['N'], 2)
                trueElim = int(1 - truePrev)
                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    vaccState=copy.deepcopy(simData['sv']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    adherenceFactors=copy.deepcopy(
                        simData['adherenceFactors']),
                    compliers=copy.deepcopy(simData['compliers']),
                    nVacc = simData['vaccCount'],
                    nChemo1 = simData['nChemo1'],
                    nChemo2 = simData['nChemo2'],
                    nSurvey = nSurvey,
                    surveyPass = surveyPass,
                    elimination = trueElim
                ))
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime,
                       nextAgeTime, nextVaccTime])

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #     # ageAtChemo=np.array(simData['ageAtChemo']),
    #     # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results


def SCH_Simulation(paramFileName, demogName, numReps=None):
    '''
    This function generates multiple simulation paths.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    numReps: int
        number of simulations;
    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''

    # initialize the parameters
    params = loadParameters(paramFileName, demogName)

    # extract the number of simulations
    if numReps is None:
        numReps = params['numReps']

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealization)(params, i) for i in range(numReps))

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalence(output, params, numReps)

    return df


def SCH_Simulation_DALY(paramFileName, demogName, numReps=None):
    '''
    This function generates multiple simulation paths.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    numReps: int
        number of simulations;
    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''

    # initialize the parameters
    params = loadParameters(paramFileName, demogName)

    # extract the number of simulations
    if numReps is None:
        numReps = params['numReps']

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealization)(params, i) for i in range(numReps))

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalenceDALYsAll(output, params, numReps)

    return df



def getCostData(results, params):
    for i in range(len(results)):
        df = pd.DataFrame(results[i])
        if i == 0:
            df1 = pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('nChemo1', df.shape[0]),
                   'draw_1':df['nChemo1']})
        else:
            df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('nChemo', df.shape[0]),
                   'draw_1':df['nChemo']}))
        df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('nChemo2', df.shape[0]),
                   'draw_1':df['nChemo2']}))
        df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('nVacc', df.shape[0]),
                   'draw_1':df['nVacc']}))
        df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('nSurvey', df.shape[0]),
                   'draw_1':df['nSurvey']}))
        df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('surveyPass', df.shape[0]),
                   'draw_1':df['surveyPass']}))
        df1 = df1.append(pd.DataFrame({'Time':df['time'], 
                   'age_start': np.repeat('None', df.shape[0]), 
                   'age_end':np.repeat('None',df.shape[0]), 
                   'intensity':np.repeat('None', df.shape[0]),
                   'species':np.repeat(params['species'], df.shape[0]),
                   'measure':np.repeat('trueElimination', df.shape[0]),
                   'draw_1':df['elimination']}))
        return df1

def SCH_Simulation_DALY_Coverage(paramFileName, demogName, 
                                 coverageFileName,
                                 coverageTextFileStorageName,
                                 numReps=None):
    '''
    This function generates multiple simulation paths.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    numReps: int
        number of simulations;
    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''
    cov = parse_coverage_input(coverageFileName,
                             coverageTextFileStorageName)
    # initialize the parameters
    params = loadParameters(paramFileName, demogName)
    params = readCoverageFile(coverageTextFileStorageName, params)
    # extract the number of simulations
    if numReps is None:
        numReps = params['numReps']

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealizationSurveyCoverage)(params, i) for i in range(numReps))

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalenceDALYsAll(output, params, numReps)

    return df



def singleSimulationDALYCoverage(params,simData,
                                 numReps=None):
    '''
    This function generates multiple simulation paths.
    Parameters
    ----------
    params 
        set of parameters for the run
    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''
   
    # extract the number of simulations
    if numReps is None:
        numReps = params['numReps']

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealizationSurveyCoveragePickle)(params, simData, i) for i in range(numReps))

    # process the output
    output = extractHostData(results)
    
    # transform the output to data frame
    df = getPrevalenceDALYsAll(output, params, numReps, Unfertilized= params['Unfertilized'])
    numAgeGroup = outputNumberInAgeGroup(results, params)
    costData = getCostData(results, params)
    df1 = pd.concat([df,numAgeGroup],ignore_index=True)
    df1 = pd.concat([df1, costData], ignore_index=True)
    return df1



def multiple_simulations(params, pickleData, simparams, i):
    print( f"==> multiple_simulations starting sim {i}" )
    start_time = time.time()
    # copy the parameters
    parameters = copy.deepcopy(params)

    # load the previous simulation results
    data = pickleData[i]

    # extract the previous simulation output
    keys = ['si', 'worms', 'freeLiving', 'demography', 'contactAgeGroupIndices', 'treatmentAgeGroupIndices']
    t = 0
    
    simData = dict((key, copy.deepcopy(data[key])) for key in keys)
    simData['sv'] = np.zeros(len(simData['si']) ,dtype = int)
    simData['attendanceRecord'] = []
    simData['ageAtChemo'] = []
    simData['adherenceFactorAtChemo'] = []
    simData['vaccCount'] = 0
    simData['nChemo1'] = 0
    simData['nChemo2'] = 0
    simData['numSurvey'] = 0
    simData['compliers'] = np.random.uniform(low=0, high=1, size=len(simData['si'])) > params['propNeverCompliers']
    simData['adherenceFactors']= np.random.uniform(low=0, high=1, size=len(simData['si']))
    # extract the previous random state
    #state = data['state']
    
    # extract the previous simulation times
    times = data['times']
    simData['demography']['birthDate'] = simData['demography']['birthDate'] - times['maxTime']
    simData['demography']['deathDate'] = simData['demography']['deathDate'] - times['maxTime']
    
    simData['contactAgeGroupIndices'] = pd.cut(x=t - simData['demography']['birthDate'], bins=parameters['contactAgeGroupBreaks'],
    labels=np.arange(0, len(parameters['contactAgeGroupBreaks']) - 1)).to_numpy()
    parameters['N'] = len(simData['si'])
    # increment the simulation times
   # parameters['maxTime'] += times['maxTime']
   
    # update the parameters
    #seed = simparams.iloc[i, 0].tolist()
    R0 = simparams.iloc[i, 1].tolist()
    k = simparams.iloc[i, 2].tolist()
    parameters['R0'] = R0
    parameters['k'] = k

    # configure the parameters
    parameters = configure(parameters)
    parameters['psi'] = getPsi(parameters)
    parameters['equiData'] = getEquilibrium(parameters)
    #parameters['moderateIntensityCount'], parameters['highIntensityCount'] = setIntensityCount(paramFileName)

    # add a simulation path
    # results = doRealizationSurveyCoveragePickle(params, simData, 1)
    # output = extractHostData(results)

    # transform the output to data frame
    df = singleSimulationDALYCoverage(parameters, simData, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print( f"==> multiple_simulations finishing sim {i}: {total_time:.3f}s" )
    return df
