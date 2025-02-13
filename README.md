# SCH Simulation Model

To run the SCH simulation model, import the `SCH_Simulation()` function from the `helsim_RUN_KK` module in the `sch_simulation` package.

The `SCH_Simulation()` function requires the following inputs:

- `paramFileName`: name of the input file with the model parameters. The available files are listed in the following table.

| paramFileName |
| :--- | 
| SCH-high_adult_burden.txt | 
| SCH-low_adult_burden.txt  | 

- `demogName`: name of the demography. The available demographies are listed in the following table.

| demogName | 
| :--- | 
| Default | 
| WHOGeneric  | 
| UgandaRural | 
| KenyaKDHS  | 
| Flat | 

- `numReps`: number of simulations. If not provided, the number of simulations is extracted from the parameters file.

The `SCH_Simulation()` function returns a data frame with the following columns: `time`, `SAC Prevalence`, 
`Adult Prevalence`, `SAC Heavy Intensity Prevalence` and `Adult Heavy Intensity Prevalence`.

The output data frame can be exported in several different formats; see `sch_results.json` for an example of the results in JSON format.

See also `sch_run.py` for an example of how to use the `SCH_Simulation()` function.

### Installation

Required: Python 3.9 or higher.

It is recommended to install the package into a specific Python
"virtual" environment, see [Install packages in a virtual environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#install-packages-in-a-virtual-environment-using-pip-and-venv).

```shell
$ git clone https://github.com/NTD-Modelling-Consortium/ntd-model-sch.git
$ pip install ntd-model-sch/
```

See the [Contributing](#Developer-install) section for instructions on
installing the model package for developers.

### Automatic tests

```
$ pipenv run pytest tests
```

The file `tests/test_single_simulation.py` runs a single SCH
simulation over 24 years.  The resulting yearly prevalence values are compared to reference values stored in `tests/data/prevalence_reference.txt`.

Reference prevalence data can be automatically updated by running:

```
pipenv run tests/test_single_simulation --update-ref
```

## Endgame inputs

There are two input files required to run endgame simulations :
-  a text file containing the parameters e.g. ascaris_scenario_3a.txt
-  a spreadsheet containing the intervention timings and coverage e.g. ascaris_coverage_scenario_3a.xlsx


### Text file
The text file must contain the following categories of parameters : 

#### Run time parameters
`repNum` : number of repetitions \
`nYears`	: number of years to run\
`outputEvents`	: what time intervals should events be outputted at 

#### Underlying biological and epidemiological parameters
`species` : the species name, e.g. ascaris\
`unfertilized` : unfertilised marker\
`k` : shape parameter of assumed negative binomial distribution of worms amongst host\
`lambda` : Eggs per gram (lambda)\
`R0` : Basic reproductive number (R0)\
`ReservoirDecayRate` : Reservoir decay rate (decay rate of eggs in the environment)\
`sigma` : Worm death rate (sigma) i.e. 1/worm_life_span, same for all development stages\
`gamma` : Exponential density dependence of parasite adult stage (gamma) N.B. fecundity parameter z = exp(-gamma)\
`k_epg`: size parameter of negative binomial for test\
`reproFuncNam` : name of function for reproduction (a string). Deterministic\
`StochSR` : Turn SR on or off in the stochastic model. Stochastic\
`nHosts`	: Size of definitive host population (N)\
`neverTreated`	: proportion never treated\
`contactAgeBreaks` : Contact age group breaks (minus sign necessary to include zero age)\
`betaValues` : Relative contact rates (Beta)\
`rhoValues`: Rho, contribution to the reservoir by contact age group

#### Post-processing parameters
`outputBreaks` : What are the output age classes? \
`highBurdenBreaks` : Three categories here \
`highBurdenValues` : Corresponding values \
`mediumThreshold` : threshold for medium intensity infection \
`heavyThreshold` : threshold for heavy intensity infection 

#### Treatment parameters 
`treatmentBreaks` : Minimum age of each treatment group (minus sign necessary to include zero age): Infants; Pre-SAC; SAC; Adults \
`drugEff`	 : Drug efficacy 
`drugEff1` : Drug efficacy of old product if more than one drug is used \
`drugEff2` : Drug efficacy of new product if more than one drug is used 

Currently the coverage and treatment information can be passed either within the .txt file or in the spreadsheet (see section below). If the coverage information is passed in the text file, the following parameters must be specified: 

`treatInterval1`	: interval between treatments in years, pre-COVID \
`treatInterval2`	: interval between treatments in years, post-COVID \
`treatStart1`	: Treatment year start, pre-COVID \
`treatStart2`	:Treatment year start, post-COVID \
`nRounds1`	: Number of treatment rounds, pre-COVID \
`nRounds2`	: Number of treatment rounds, post-COVID \
`coverage1`	: Coverages pre-COVID: Infants; Pre-SAC; SAC; Adults 
`coverage2`	: Coverages post-COVID: Infants; Pre-SAC; SAC; Adults 

#### Vaccine parameters
`V1sigma` : impact of vaccine on worm death rate. Assume worm death rate is v1sigma \
`V2lambda` : impact of vaccine on eggs per gram   Fraction of eggs produced when vaccinated \
`v3betaValues`	: impact of vaccine on contact rates  Assume contact rate under vaccination is times v3\
`VaccDecayRate`: vacc decay rate. rate of vaccine decay = 1/duration of vaccine   A vector with value 0 in state 1 and the vacc decay rate for state 2 

Currently the vaccine information can be passed either within the .txt file or in the spreadsheet (see section below). If the vaccine information is passed in the text file, the following parameters must be specified: 

`VaccTreatmentBreaks` : age range of the vaccinated group. These are the lower bounds of ranges with width 1, they must be more than one year part \
`VaccCoverage`	  : Vaccine coverage of the age group \
`VaccTreatStart` : Vaccine administration year start\
`nRoundsVacc`	: number of vaccine rounds \
`treatIntervalVacc` : time interval between vaccine rounds

#### Survey parameters
`timeToFirstSurvey` : time of the first survey after simulation starts\
`timeToNextSurvey` : time between surveys\
`surveyThreshold` : prevalence threshold for passing survey\
`sampleSizeOne` : sample size for the first survey\
`sampleSizeTwo` : sample size for the second survey\
`nSamples` : how many samples to take from each individual\
`minSurveyAge` : minimum survey age\
`maxSurveyAge` : maximum survey age 

### Spreadsheet

The treatment and vaccination timings and coverages can be passed in a .xlsx file. The spreadsheet consists of two tabs:
- PlatformCoverage
- MarketShare

The PlatformCoverage tab must contain the following columns:
- InterventionType : the type of intervention, Treatment, Vaccine
- min age : the minimum age of the intervention recipients 
- max age : the maximum age of the intervention recipients 
- Then one column for each time point of application of the intervention, containing the coverage percentage.

Example row:

| Country/Region | Intervention Type| Platform Type | Cohort (if not total pop in country/region) |min age | max age | 2018 | 2018.5 |2019 |2019.5 | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| All | Treatment | Campaign | |  5 | 15 | 0.6 |  0| 0.6| 0|

The columns “Country/Region”, “Platform Type” and “Platform” are currently optional.

The MarketShare tab must contain the following columns: 
- Product : one of ‘New Product A’ or ‘Old Product B (SOC)’
- Then one column for each time point of application of mass treatment, containing market share percentage of ‘New Product A’ or ‘Old Product B (SOC)’

Example row:

| Country/Region | Platform| Product | 2018 | 2018.5 |2019 |2019.5 | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| DRC| MDA | New Product A | |  | 0.5 | 1|  1|
| DRC| MDA |Old Product B (SOC) | 1 | 1 | 0.5 | |  |

The columns “Country/Region” and “Platform” are currently optional.

### Contributing

#### Developer install

Developers wishing to contribute can install the package in "editable"
mode, in order to make changes to the source code available in the
environment right away:

```shell
$ pip install --editable ntd-model-sch/[dev]
```

See [Development Mode (a.k.a. “Editable
Installs”)](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

The suffix `[dev]` enables installation of the developement
utilities `pytest` and `ruff`.

#### Running the tests and linter

Before committing, and even more importantly before pushing new commits
to the repository, please ensure that the automatic tests pass and that the `ruff` linter raises no errors. You can do so with

```shell
# From the project's root directory
$ pytest tests/
$ ruff check sch_simulation/
```

#### Code formatting

Code pushed to the repository is expected to be formatted according to
the rules enforced by the `ruff format` tool, version 0.4.* with
default settings. To automatically format the files your changed by
running

```shell
ruff format sch_simulation/path/to/file1 /sch_simulation/path/to/file2 ...
```

or simply

```shell
# Format all *.py files in sch_simulation directory
ruff format sch_simulation
```