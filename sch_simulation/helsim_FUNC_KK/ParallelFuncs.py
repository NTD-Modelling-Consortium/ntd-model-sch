import numpy as np

from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters


def epgPerPerson(x: np.ndarray, params: Parameters) -> np.ndarray:

    """
    This function calculates the total eggs per gram as
    a function of the mean worm burden.

    Parameters
    ----------
    x: float
        array of mean worm burdens;

    params: dict
        dictionary containing the parameter names and values;
    """
    return (
        params.lambda_egg
        * x
        * params.z
        * (1 + x * (1 - params.z) / params.k) ** (-params.k - 1)
    )


def fertilityFunc(x: np.ndarray, params: Parameters) -> np.ndarray:

    """
    This function calculates the multiplicative fertility correction factor
    to be applied to the mean eggs per person function.

    Parameters
    ----------
    x: float
        array of mean worm burdens;

    params: dict
        dictionary containing the parameter names and values;
    """

    a = 1 + x * (1 - params.z) / params.k
    b = 1 + 2 * x / params.k - params.z * x / params.k

    return 1 - (a / b) ** (params.k + 1)


def monogFertilityFuncApprox(x: float, params: Parameters):

    """
    This function calculates the fertility factor for monogamously mating worms.

    Parameters
    ----------

    x: float
        mean worm burden;

    params: dict
        dictionary containing the parameter names and values;
    """
    assert params.monogParams is not None
    if x > 25 * params.k:

        return 1 - params.monogParams.c_k / np.sqrt(x)

    else:

        g = x / (x + params.k)
        integrand = (1 - params.monogParams.cosTheta) * (
            1 + float(g) * params.monogParams.cosTheta
        ) ** (-1 - float(params.k))
        integral = np.mean(integrand)

        return 1 - (1 - g) ** (1 + params.k) * integral


def epgMonog(x: np.ndarray, params: Parameters) -> np.ndarray:

    """
    This function calculates the generation of eggs with monogamous
    reproduction taken into account.

    Parameters
    ----------
    x: float
        array of mean worm burdens;

    params: dict
        dictionary containing the parameter names and values;
    """
    vectorized = np.array([monogFertilityFuncApprox(i, params) for i in x])
    return epgPerPerson(x, params) * vectorized


def epgFertility(x: np.ndarray, params: Parameters) -> np.ndarray:

    """
    This function calculates the generation of eggs with
    sexual reproduction taken into account.

    Parameters
    ----------
    x: float
        array of mean worm burdens;

    params: dict
        dictionary containing the parameter names and values;
    """

    return epgPerPerson(x, params) * fertilityFunc(x, params)


mapper = {
    "epgPerPerson": epgPerPerson,
    "fertilityFunc": fertilityFunc,
    "epgMonog": epgMonog,
    "epgFertility": epgFertility,
}
