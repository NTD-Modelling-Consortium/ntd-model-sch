source("../../sch_simulation/amis_integration/reticulate_util.R")

test_that("Getting virtual environement from here gets virtual env path", {
  venv_path <- get_venv()
  expect_match(venv_path, ".*virtualenvs/ntd-model-sch.*")
})

test_that("Getting a virtual env from outside this project gets null", {
  setwd("/") # leave the project so no appropriate virtual environement
  # The command generates a warning since
  # the command exits with non-zero exit status
  suppressWarnings({
    expect_error(get_venv(), "No Python virtual enviroment found.*")
  })
})
