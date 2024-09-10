import warnings
from typing import List, Tuple
import math
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray

from sch_simulation.helsim_FUNC_KK.helsim_structures import (
    Parameters,
    ProcResult,
    Result,
    SDEquilibrium,
)
from sch_simulation.helsim_FUNC_KK.utils import getSetOfEggCounts, getSetOfEggCountsv2, POC_CCA_test, PCR_test

warnings.filterwarnings("ignore")

np.seterr(divide="ignore")

# When running a single simulation the output is always put in a column called
# draw_1.
OUTPUT_COLUMN_NAME = "draw_1"

def extractHostData(results: List[List[Result]]) -> List[ProcResult]:

    """
    This function is used for processing results the raw simulation results.
    Parameters
    ----------
    results: List[List[Result]]
        raw simulation output;
    Returns
    -------
    output: List[ProcResult]
        processed simulation output;
    """

    output = []

    for result in results:

        output.append(
            ProcResult(
                vaccState = np.array(
                    [result[i].vaccState for i in range(len(result))]
                ).T,
                wormsOverTime=np.array(
                    [result[i].worms.total for i in range(len(result))]
                ).T,
                femaleWormsOverTime=np.array(
                    [result[i].worms.female for i in range(len(result) )]
                ).T,
                # freeLiving=np.array([result[i]['freeLiving'] for i in range(len(results[0]) - 1)]),
                ages=np.array(
                    [
                        result[i].time - result[i].hosts.birthDate
                        for i in range(len(result) )
                    ]
                ).T,
                # adherenceFactors=np.array([result[i]['adherenceFactors'] for i in range(len(results[0]) - 1)]).T,
                # compliers=np.array([result[i]['compliers'] for i in range(len(results[0]) - 1)]).T,
                # totalPop=len(result[0]['worms']['total']),
                timePoints=np.array(
                    [np.array(result[i].time) for i in range(len(result) )]
                ),
                prevalence = np.array(
                    [np.array(result[i].prevalence) for i in range(len(result) )]
                ),
                # attendanceRecord=result[-1]['attendanceRecord'],
                # ageAtChemo=result[-1]['ageAtChemo'],
                # finalFreeLiving=result[-2]['freeLiving'],
                # adherenceFactorAtChemo=result[-1]['adherenceFactorAtChemo']
                # sex_id = np.array([result[i]['sex_id'] for i in range(len(results[0]) - 1)]).T
            )
        )

    return output


def getVillageMeanCountsByHost(
    villageList: ProcResult,
    timeIndex: int,
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    surveyType: str = "KK2"
) -> NDArray[np.float_]:
    """
    This function returns the mean egg count across readings by host
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
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
    """

    meanEggsByHost = (
        getSetOfEggCounts(
            villageList.wormsOverTime[:, timeIndex],
            villageList.femaleWormsOverTime[:, timeIndex],
            villageList.vaccState[:, timeIndex],
            params,
            Unfertilized,
            nSamples,
            surveyType
        )
        / nSamples
    )


    return meanEggsByHost


def getVillageMeanCountsByHostv2(
    villageList: ProcResult,
    timeIndex: int,
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    surveyType: str = "KK2"
) -> NDArray[np.float_]:
    """
    This function returns the mean egg count across readings by host
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
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
    """

    meanEggsByHost = (
        getSetOfEggCountsv2(
            villageList.wormsOverTime[:, timeIndex],
            villageList.femaleWormsOverTime[:, timeIndex],
            villageList.vaccState[:, timeIndex],
            params,
            Unfertilized,
            nSamples,
            surveyType
        )
        / nSamples
    )


    return meanEggsByHost



def getVillagePOCCCAByHost(
    villageList: ProcResult,
    timeIndex: int,
    params: Parameters,
    Unfertilized: bool,
    surveyType: str,
    nSamples: int = 2,
) -> NDArray[np.float_]:
    """
    This function returns the mean egg count across readings by host
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
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
    """

    mean_POCCCA = POC_CCA_test(
        villageList.wormsOverTime[:, timeIndex],
        params)
    
    for i in range(1, nSamples):
        mean_POCCCA += POC_CCA_test(
            villageList.wormsOverTime[:, timeIndex],
            params)

    return mean_POCCCA / nSamples


def getAgeCatSampledPrevByVillage(
    villageList: ProcResult,
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
    surveyType: str = 'KK2'
) -> float:

    """
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: NDArray[int]
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    meanEggCounts = getVillageMeanCountsByHost(
        villageList, timeIndex, params, Unfertilized, nSamples
    )

    ageGroups = (
        np.digitize(
            villageList.ages[:, timeIndex], np.append(-10, np.append(ageBand, 150))
        )
        
    )

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(
            a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False
        )

    else:
        mySample = np.random.choice(
            a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True
        )

    return np.sum(nSamples * mySample > 0.9) / villageSampleSize


def getAgeCatSampledPrevByVillageAll(
    villageList: ProcResult,
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    surveyType: str = 'KK2',
    nSamples: int = 2,
    villageSampleSize=100,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:

    """
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """
   
    if ((surveyType == "KK1") | (surveyType == "KK2")):
        meanEggCounts = getVillageMeanCountsByHost(
            villageList, timeIndex, params, Unfertilized, nSamples, surveyType
        )
        ages = villageList.ages[:, timeIndex]
    
        currentAges = np.where(np.logical_and(ages >= ageBand[0], ages < ageBand[1]))
        currentAgeGroupMeanEggCounts = meanEggCounts[currentAges]
    
        is_empty = currentAgeGroupMeanEggCounts.size == 0
        
        if is_empty:
            infected = np.nan
            low = np.nan
            medium = np.nan
            heavy = np.nan
            meanEggs = np.nan
        else:
    
            infected = np.sum(currentAgeGroupMeanEggCounts>0)/len(currentAgeGroupMeanEggCounts)
            #meanEggCounts = getVillageMeanCountsByHost(
            #     villageList, timeIndex, params, Unfertilized, nSamples, surveyType
            #)
            ages = villageList.ages[:, timeIndex]
        
            currentAges = np.where(np.logical_and(ages >= ageBand[0], ages < ageBand[1]))
            currentAgeGroupMeanEggCounts = meanEggCounts[currentAges]
        
            is_empty = currentAgeGroupMeanEggCounts.size == 0
            # print("mod thr4esh = ", params.mediumThreshold)
            # print("heav thr4esh = ", params.heavyThreshold)
            medium = (
                np.sum(
                    (currentAgeGroupMeanEggCounts >= params.mediumThreshold)
                    & (currentAgeGroupMeanEggCounts <= params.heavyThreshold)
                )
                /len(currentAgeGroupMeanEggCounts)
            )
            
            heavy = np.sum(currentAgeGroupMeanEggCounts > params.heavyThreshold) / len(currentAgeGroupMeanEggCounts)
            # print(currentAgeGroupMeanEggCounts)
            # print("heavy infs = " , heavy)
            low = infected - (medium + heavy)
            meanEggs = np.mean(currentAgeGroupMeanEggCounts)
        return (
            np.array(infected),
            np.array(low),
            np.array(medium),
            np.array(heavy),
            np.array(len(currentAgeGroupMeanEggCounts)),
            np.array(meanEggs)
        )
    
    
    
    if surveyType == "POC-CCA":
        meanPOCCCA = getVillagePOCCCAByHost(
            villageList, timeIndex, params, Unfertilized, surveyType, nSamples
        ) 
    
        ages = villageList.ages[:, timeIndex]
    
        currentAges = np.where(np.logical_and(ages >= ageBand[0], ages < ageBand[1]))
        currentAgeGroupMeanEggCounts = meanPOCCCA[currentAges]
    
        is_empty = currentAgeGroupMeanEggCounts.size == 0
        
        if is_empty:
            infected = np.nan
            low = np.nan
            medium = np.nan
            heavy = np.nan
            meanPOCmeasure = np.nan
        else:
    
            infected = np.sum(currentAgeGroupMeanEggCounts>params.POC_CCA_thresholds[0])/len(currentAgeGroupMeanEggCounts)
     
            medium = (
                np.sum(
                    (currentAgeGroupMeanEggCounts >= params.POC_CCA_thresholds[1])
                    & (currentAgeGroupMeanEggCounts <= params.POC_CCA_thresholds[2])
                )
                /len(currentAgeGroupMeanEggCounts)
            )
    
            heavy = np.sum(currentAgeGroupMeanEggCounts > params.POC_CCA_thresholds[2]) / len(currentAgeGroupMeanEggCounts)
    
            low = infected - (medium + heavy)
            meanPOCmeasure = np.round(np.mean(currentAgeGroupMeanEggCounts))    
    
    
        return (
            np.array(infected),
            np.array(low),
            np.array(medium),
            np.array(heavy),
            np.array(len(currentAgeGroupMeanEggCounts)),
            np.array(meanPOCmeasure)
        )


def getAgeCatSampledPrevByVillageAllPOCCCA(
    villageList: ProcResult,
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize=100,
    surveyType: str = 'POC-CCA'
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:

    """
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    meanEggCounts = getVillageMeanCountsByHost(
        villageList, timeIndex, params, Unfertilized, surveyType, nSamples
    )
    ages = villageList.ages[:, timeIndex]
    #ageGroups = (
    #    np.digitize(
    #        villageList.ages[:, timeIndex], np.append(-10, np.append(ageBand, 150))
    #    )
    #    - 1
    #)
    currentAges = np.where(np.logical_and(ages >= ageBand[0], ages < ageBand[1]))
    currentAgeGroupMeanEggCounts = meanEggCounts[currentAges]

    is_empty = currentAgeGroupMeanEggCounts.size == 0
    
    if is_empty:
        infected = np.nan
        low = np.nan
        medium = np.nan
        heavy = np.nan
        meanEggs = np.nan
    else:

        infected = np.sum(currentAgeGroupMeanEggCounts>0)/len(currentAgeGroupMeanEggCounts)
 
        medium = (
            np.sum(
                (currentAgeGroupMeanEggCounts >= params.mediumThreshold)
                & (currentAgeGroupMeanEggCounts <= params.heavyThreshold)
            )
            /len(currentAgeGroupMeanEggCounts)
        )

        heavy = np.sum(currentAgeGroupMeanEggCounts > params.heavyThreshold) / len(currentAgeGroupMeanEggCounts)

        low = infected - (medium + heavy)
        meanEggs = np.mean(currentAgeGroupMeanEggCounts)
    return (
        np.array(infected),
        np.array(low),
        np.array(medium),
        np.array(heavy),
        np.array(len(currentAgeGroupMeanEggCounts)),
        np.array(meanEggs)
    )



def getAgeCatSampledPrevHeavyBurdenByVillage(
    villageList: ProcResult,
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
    surveyType: int = 'KK2'
) -> float:
    """
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.
    Parameters
    ----------
    villageList: ProcResult
        processed simulation output for a given iteration;
    timeIndex: int
        selected time point index;
    ageBand: NDArray[int]
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    meanEggCounts = getVillageMeanCountsByHost(
        villageList, timeIndex, params, Unfertilized,  nSamples
    )
    ageGroups = (
        np.digitize(
            villageList.ages[:, timeIndex], np.append(-10, np.append(ageBand, 150))
        )
        
    )

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(
            a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False
        )

    else:
        mySample = np.random.choice(
            a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True
        )

    return np.sum(mySample > params.heavyThreshold) / villageSampleSize


def getSampledDetectedPrevByVillageAll(
    hostData: List[ProcResult],
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    surveyType: str, 
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> List[Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]]:

    """
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: NDArray[int]
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    return [
        getAgeCatSampledPrevByVillageAll(
            villageList,
            timeIndex,
            ageBand,
            params,
            Unfertilized,
            surveyType, 
            nSamples,
            villageSampleSize,
        )
        for villageList in hostData
    ]


def getBurdens(
    hostData: List[ProcResult],
    params: Parameters,
    numReps: int,
    ageBand: NDArray[np.int_],
    Unfertilized: bool,
    surveyType: str, 
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> Tuple[
    NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]
]:

    results = np.empty((0, numReps))
    low_results = np.empty((0, numReps))
    medium_results = np.empty((0, numReps))
    heavy_results = np.empty((0, numReps))
    mean_eggs = np.empty((0, numReps))
    for t in range(len(hostData[0].timePoints)):  # loop over time points
        # calculate burdens using the same sample
        #newrow = np.array(
        #  getSampledDetectedPrevByVillageAll(
        #        hostData, t, ageBand, params, Unfertilized, surveyType, nSamples, villageSampleSize
        #    )
        #)
        
        newrow = np.array(
            getSampledDetectedPrevByVillageAll(
                hostData, t, ageBand, params, Unfertilized, surveyType, 1, villageSampleSize
            )
        )
        newrowinfected = newrow[:, 0]
        newrowlow = newrow[:, 1]
        newrowmedium = newrow[:, 2]
        newrowheavy = newrow[:, 3]
        newroweggs = newrow[:, 5]
        # append row
        results = np.vstack([results, newrowinfected])
        low_results = np.vstack([low_results, newrowlow])
        medium_results = np.vstack([medium_results, newrowmedium])
        heavy_results = np.vstack([heavy_results, newrowheavy])
        mean_eggs = np.vstack([mean_eggs, newroweggs])
    # calculate proportion across number of repetitions
    prevalence: NDArray[np.float_] = np.sum(results, axis=1) / numReps
    low_prevalence: NDArray[np.float_] = np.sum(low_results, axis=1) / numReps
    medium_prevalence: NDArray[np.float_] = np.sum(medium_results, axis=1) / numReps
    heavy_prevalence: NDArray[np.float_] = np.sum(heavy_results, axis=1) / numReps
    mean_egg_res: NDArray[np.float_] = np.sum(mean_eggs, axis=1) / numReps
    
    return prevalence, low_prevalence, medium_prevalence, heavy_prevalence, mean_egg_res


def getSampledDetectedPrevByVillage(
    hostData: List[ProcResult],
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
    surveyType: str = 'KK2'
) -> NDArray[np.float_]:

    """
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: int
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    return np.array(
        [
            getAgeCatSampledPrevByVillage(
                villageList,
                timeIndex,
                ageBand,
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for villageList in hostData
        ]
    )


def getSampledDetectedPrevHeavyBurdenByVillage(
    hostData: List[ProcResult],
    timeIndex: int,
    ageBand: NDArray[np.int_],
    params: Parameters,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> NDArray[np.float_]:
    """
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    timeIndex: int
        selected time point index;
    ageBand: NDArray[int]
        array with age group boundaries;
    params: Parameters
        dataclass containing the parameter names and values;
    nSamples: int
        number of samples;
    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;
    villageSampleSize: int;
        village sample size fraction;
    Returns
    -------
    sampled worm prevalence;
    """

    return np.array(
        [
            getAgeCatSampledPrevHeavyBurdenByVillage(
                villageList,
                timeIndex,
                ageBand,
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for villageList in hostData
        ]
    )


def getPrevalence(
    hostData: List[ProcResult],
    params: Parameters,
    numReps: int,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
    surveyType: str = 'KK2'
) -> pd.DataFrame:

    """
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    params: Parameters
        dataclass containing the parameter names and values;
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
    """

    sac_results = np.array(
        [
            getSampledDetectedPrevByVillage(
                hostData,
                t,
                np.array([5, 15]),
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for t in range(len(hostData[0].timePoints))
        ]
    )

    adult_results = np.array(
        [
            getSampledDetectedPrevByVillage(
                hostData,
                t,
                np.array([16, 80]),
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for t in range(len(hostData[0].timePoints))
        ]
    )

    sac_heavy_results = np.array(
        [
            getSampledDetectedPrevHeavyBurdenByVillage(
                hostData,
                t,
                np.array([5, 15]),
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for t in range(len(hostData[0].timePoints))
        ]
    )

    adult_heavy_results = np.array(
        [
            getSampledDetectedPrevHeavyBurdenByVillage(
                hostData,
                t,
                np.array([16, 80]),
                params,
                Unfertilized,
                nSamples,
                villageSampleSize,
            )
            for t in range(len(hostData[0].timePoints))
        ]
    )

    sac_prevalence = np.sum(sac_results, axis=1) / numReps
    adult_prevalence = np.sum(adult_results, axis=1) / numReps

    sac_heavy_prevalence = np.sum(sac_heavy_results, axis=1) / numReps
    adult_heavy_prevalence = np.sum(adult_heavy_results, axis=1) / numReps

    df = pd.DataFrame(
        {
            "Time": hostData[0].timePoints,
            "SAC Prevalence": sac_prevalence,
            "Adult Prevalence": adult_prevalence,
            "SAC Heavy Intensity Prevalence": sac_heavy_prevalence,
            "Adult Heavy Intensity Prevalence": adult_heavy_prevalence,
        }
    )

    #df = df[(df["Time"] >= 50) & (df["Time"] <= 64)]
    #df["Time"] = df["Time"] - 50

    return df


def getPrevalenceDALYs(
    hostData: List[ProcResult],
    params: Parameters,
    numReps: int,
    Unfertilized: bool,
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> pd.DataFrame:
    """
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    params: Parameters
        dataclass containing the parameter names and values;
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
    """

    # under 4s
    (
        ufour_prevalence,
        ufour_low_prevalence,
        ufour_medium_prevalence,
        ufour_heavy_prevalence,
    ) = getBurdens(
        hostData,
        params,
        numReps,
        np.array([0, 4]),
        Unfertilized,
        nSamples=2,
        villageSampleSize=100,
    )

    # adults
    (
        adult_prevalence,
        adult_low_prevalence,
        adult_medium_prevalence,
        adult_heavy_prevalence,
    ) = getBurdens(
        hostData,
        params,
        numReps,
        np.array([5, 80]),
        Unfertilized,
        nSamples=2,
        villageSampleSize=100,
    )

    # all individuals
    (
        all_prevalence,
        all_low_prevalence,
        all_medium_prevalence,
        all_heavy_prevalence,
    ) = getBurdens(
        hostData,
        params,
        numReps,
        np.array([0, 80]),
        Unfertilized,
        nSamples=2,
        villageSampleSize=100,
    )

    df = pd.DataFrame(
        {
            "Time": hostData[0].timePoints,
            "Prevalence": all_prevalence,
            "Low Intensity Prevalence": all_low_prevalence,
            "Medium Intensity Prevalence": all_medium_prevalence,
            "Heavy Intensity Prevalence": all_heavy_prevalence,
            "Under four Prevalence": ufour_prevalence,
            "Under four Low Intensity Prevalence": ufour_low_prevalence,
            "Under four Medium Intensity Prevalence": ufour_medium_prevalence,
            "Under four Heavy Intensity Prevalence": ufour_heavy_prevalence,
            "Adult Prevalence": adult_prevalence,
            "Adult Low Intensity Prevalence": adult_low_prevalence,
            "Adult Medium Intensity Prevalence": adult_medium_prevalence,
            "Adult Heavy Intensity Prevalence": adult_heavy_prevalence,
        }
    )

    df = df[(df["Time"] >= 50) & (df["Time"] <= 64)]
    df["Time"] = df["Time"] - 50

    return df


def getPrevalenceDALYsAll(
    hostData: List[ProcResult],
    params: Parameters,
    numReps: int,
    Unfertilized: bool,
    surveyType: str,
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> pd.DataFrame:
    """
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    params: Parameters
        dataclass containing the parameter names and values;
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
    """


    df = None
    for i in range(0, int(params.maxHostAge)):  # loop over yearly age bins
  
        prevalence, low_prevalence, moderate_prevalence, heavy_prevalence, meanEggs = getBurdens(
                hostData,
                params,
                numReps,
                np.array([i, i + 1]),
                Unfertilized,
                surveyType, 
                nSamples=nSamples,
                villageSampleSize=100,
            )
        age_start = i
        age_end = i + 1
            # year = hostData[0]['timePoints']

        if i == 0:
            df = pd.DataFrame(
                {
                    "Time": hostData[0].timePoints,
                    "age_start": np.repeat(age_start, len(low_prevalence)),
                    "age_end": np.repeat(age_end, len(low_prevalence)),
                    "intensity": np.repeat("light", len(low_prevalence)),
                    "species": np.repeat(params.species, len(low_prevalence)),
                    "measure": np.repeat("prevalence", len(low_prevalence)),
                    OUTPUT_COLUMN_NAME: np.round(low_prevalence,4),
                }
            )

        else:
            assert df is not None
            newrows = pd.DataFrame(
                    {
                        "Time": hostData[0].timePoints,
                        "age_start": np.repeat(age_start, len(low_prevalence)),
                        "age_end": np.repeat(age_end, len(low_prevalence)),
                        "intensity": np.repeat("light", len(low_prevalence)),
                        "species": np.repeat(params.species, len(low_prevalence)),
                        "measure": np.repeat("prevalence", len(low_prevalence)),
                        OUTPUT_COLUMN_NAME: np.round(low_prevalence,4),
                    }
                )
            df = pd.concat([df, newrows], ignore_index = True)
            

        newrows = pd.DataFrame(
                {
                    "Time": hostData[0].timePoints,
                    "age_start": np.repeat(age_start, len(low_prevalence)),
                    "age_end": np.repeat(age_end, len(low_prevalence)),
                    "intensity": np.repeat("moderate", len(low_prevalence)),
                    "species": np.repeat(params.species, len(low_prevalence)),
                    "measure": np.repeat("prevalence", len(low_prevalence)),
                    OUTPUT_COLUMN_NAME: np.round(moderate_prevalence,4),
                }
            )
        df = pd.concat([df, newrows], ignore_index = True)

        newrows = pd.DataFrame(
                {
                    "Time": hostData[0].timePoints,
                    "age_start": np.repeat(age_start, len(low_prevalence)),
                    "age_end": np.repeat(age_end, len(low_prevalence)),
                    "intensity": np.repeat("heavy", len(low_prevalence)),
                    "species": np.repeat(params.species, len(low_prevalence)),
                    "measure": np.repeat("prevalence", len(low_prevalence)),
                    OUTPUT_COLUMN_NAME: np.round(heavy_prevalence,4),
                }
            )
        df = pd.concat([df, newrows], ignore_index = True)
        
        # newrows = pd.DataFrame(
        #         {
        #             "Time": hostData[0].timePoints,
        #             "age_start": np.repeat(age_start, len(low_prevalence)),
        #             "age_end": np.repeat(age_end, len(low_prevalence)),
        #             "intensity": np.repeat("None", len(low_prevalence)),
        #             "species": np.repeat(params.species, len(low_prevalence)),
        #             "measure": np.repeat("meanEggs", len(low_prevalence)),
        #             OUTPUT_COLUMN_NAME: np.round(meanEggs,4),
        #         }
        #     )
        # df = pd.concat([df, newrows], ignore_index = True)

     

    return df


def getIncidence(results: List[List[Result]], params: Parameters) -> pd.DataFrame:

    for i in range(len(results[0])):
        d = results[0][i]
        value, _ = np.histogram(
                d.incidenceAges,
                bins=np.arange(0, params.maxHostAge + 1),
            )
        
        newrows = pd.DataFrame(
                {
                    "Time": np.repeat(d.time, len(value)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(value)),
                    "species": np.repeat(params.species, len(value)),
                    "measure": np.repeat("Incidence", len(value)),
                    OUTPUT_COLUMN_NAME: value,
                }
            )
        if i == 0:
            incidence = newrows
        else:
            assert incidence is not None
            
            incidence = pd.concat([incidence, newrows], ignore_index = True)

    return incidence




def getPrevalenceWholePop(
    hostData: List[ProcResult],
    params: Parameters,
    numReps: int,
    Unfertilized: bool,
    surveyType: str,
    nSamples: int = 2,
    villageSampleSize: int = 100,
) -> pd.DataFrame:
    """
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.
    Parameters
    ----------
    hostData: List[ProcResult]
        processed simulation output;
    params: Parameters
        dataclass containing the parameter names and values;
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
    """


    df = None
    prevalence, _, _, _, _ = getBurdens(
                hostData,
                params,
                numReps,
                np.array([0, int(params.maxHostAge) + 1]),
                Unfertilized,
                surveyType, 
                nSamples=nSamples,
                villageSampleSize=100,
            )


    df = pd.DataFrame(
                {
                    "Time": hostData[0].timePoints,
                    "age_start": np.repeat("None", len(prevalence)),
                    "age_end": np.repeat("None", len(prevalence)),
                    "intensity": np.repeat("All", len(prevalence)),
                    "species": np.repeat(params.species, len(prevalence)),
                    "measure": np.repeat("Estimated population prevalence", len(prevalence)),
                    OUTPUT_COLUMN_NAME: np.round(prevalence,4),
                }
            )
     

    return df


def outputNumberInAgeGroup(
    results: List[List[Result]], params: Parameters
) -> pd.DataFrame:
    assert params.maxHostAge is not None
    numEachAgeGroup = None
    for i in range(len(results[0])):
        d = results[0][i]
        ages = d.time - d.hosts.birthDate
        ages1 = list(ages.astype(int))
        age_counts = []
        for j in range(int(params.maxHostAge)):
            age_counts.append(ages1.count(j))
        newrows = pd.DataFrame(
                {
                    "Time": np.repeat(d.time, len(age_counts)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(age_counts)),
                    "species": np.repeat(params.species, len(age_counts)),
                    "measure": np.repeat("number", len(age_counts)),
                    OUTPUT_COLUMN_NAME: age_counts,
                }
            )
        if i == 0:
            numEachAgeGroup = newrows
        else:
            assert numEachAgeGroup is not None
            
            numEachAgeGroup = pd.concat([numEachAgeGroup, newrows], ignore_index = True)

    return numEachAgeGroup


def outputNumberSurveyedAgeGroup(
    SD: SDEquilibrium, params: Parameters
) -> pd.DataFrame:
    assert params.maxHostAge is not None
    d = None
    count = 0
    for key, value in SD.n_surveys.items():
        t = math.floor(float(key.split(",")[0]))
        measure = str(key.split(",")[1])
        newrows = pd.DataFrame(
                {
                    "Time": np.repeat(t, len(value)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(value)),
                    "species": np.repeat(params.species, len(value)),
                    "measure": np.repeat(measure, len(value)),
                    OUTPUT_COLUMN_NAME: value,
                }
            )
        if count == 0:
            count = 1
            d = newrows
        else:
            assert d is not None
            d = pd.concat([d, newrows], ignore_index = True)
        
        v2 = value/SD.n_surveys_population[key]
        newrows2 = pd.DataFrame(
                {
                    "Time": np.repeat(t, len(value)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(value)),
                    "species": np.repeat(params.species, len(value)),
                    "measure": np.repeat("survey coverage", len(value)),
                    OUTPUT_COLUMN_NAME: v2,
                }
            )
        
        d = pd.concat([d, newrows2], ignore_index = True)


    return d



def outputNumberTreatmentAgeGroup(
    SD: SDEquilibrium, params: Parameters
) -> pd.DataFrame:
    assert params.maxHostAge is not None
    d = None
    count = 0
    for key, value in SD.n_treatments.items():
        t = math.floor(float(key.split(",")[0]))
        measure = str(key.split(",")[1])
        newrows = pd.DataFrame(
                {
                    "Time": np.repeat(t, len(value)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(value)),
                    "species": np.repeat(params.species, len(value)),
                    "measure": np.repeat(measure, len(value)),
                    OUTPUT_COLUMN_NAME: value,
                }
            )
        if count == 0:
            count = 1
            d = newrows
        else:
            assert d is not None
            d = pd.concat([d, newrows], ignore_index = True)
        
        v2 = value/SD.n_treatments_population[key]
        m1 = measure + " coverage"
        newrows2 = pd.DataFrame(
                {
                    "Time": np.repeat(t, len(value)),
                    "age_start": range(int(params.maxHostAge)),
                    "age_end": range(1, 1 + int(params.maxHostAge)),
                    "intensity": np.repeat("None", len(value)),
                    "species": np.repeat(params.species, len(value)),
                    "measure": np.repeat(m1, len(value)),
                    OUTPUT_COLUMN_NAME: v2,
                }
            )
        
        d = pd.concat([d, newrows2], ignore_index = True)

    return d
