AMIS Integration
================

Provides the relevant wrapper code to be able to call the SCH Simulation from 
the AMIS loop to compute fitted parameters. 

# Usage

## Prerequistes

 * R
 * Renv 

## Installation

Make sure you've set up the Python virtual environment for the
sch_simulation package (see main [README](../../README.md)).

1. Launch a R session `R`
2. Create a R virtual environment `renv::init()`
3. Install dependencies `renv::restore()`

## Running

```bash
$ Rscript sth_fitting.R
```

## Upgrading AMIS

AMIS is locked to specific commit in the renv.lock file. 

If the version on the repo is unchanged, then using `renv::update` won't work. 

Instead, you must remove and re-add it:

```
> renv::remove("AMISforInfectiousDiseases")
> renv::install("evandrokonzen/AMISforInfectiousDiseases-dev")
> renv::snapshot()
```