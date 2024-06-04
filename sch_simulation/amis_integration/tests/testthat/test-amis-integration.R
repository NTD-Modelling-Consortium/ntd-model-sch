setwd("../..")

source("amis_integration.R")

test_that("Running the model should give us some results", {
    sch_simulation <- get_amis_integration_package()

    fixed_parameters <- sch_simulation$FixedParameters(
        # the higher the value of N, the more consistent the results will be
        # though the longer the simulation will take
        number_hosts = 10L,
        # no intervention
        coverage_file_name = "mansoni_coverage_scenario_0.xlsx",
        demography_name = "UgandaRural",
        # cset the survey type to Kato Katz with duplicate slide
        survey_type = "KK2",
        parameter_file_name = "mansoni_params.txt",
        coverage_text_file_storage_name = "Man_MDA_vacc.txt",
        # the following number dictates the number of events (e.g. worm deaths)
        # we allow to happen before updating other parts of the model
        # the higher this number the faster the simulation
        # (though there is a check so that there can't be too many events at once)
        # the higher the number the greater the potential for
        # errors in the model accruing.
        # 5 is a reasonable level of compromise for speed and errors, but using
        # a lower value such as 3 is also quite good
        min_multiplier = 5L
    )

    # Example prevalence map, with two locations, both with prevalence of 0.5
    prevalence_map <- matrix(c(0.5, 0.5), ncol = 1)

    tranmission_model <- build_transmission_model(prevalence_map, fixed_parameters)
    result <- tranmission_model(c(1, 2), matrix(c(3, 3, 0.04, 0.04), ncol = 2), 1)
    expect_equal(result, matrix(c(0.1, 0.1), ncol = 1), tolerance = 0.5)
})
