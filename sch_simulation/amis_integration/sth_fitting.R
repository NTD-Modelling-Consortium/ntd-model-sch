source("amis_integration.R")

args <- commandArgs(trailingOnly=TRUE)
num_cores_to_use <- parallel::detectCores()
if(length(args) == 1) {
    num_cores_to_use <- as.integer(args)
}

print(paste0('Using ', num_cores_to_use, ' cores'))

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

# Example prevalence map, with two locations, fitting to two time times
# Both locations start at 0.031, and the second time point is 0.021
prevalence_map <- vector("list", 2)
prevalence_map[[1]]$data <- matrix(c(0.031, 0.031))
prevalence_map[[2]]$data <- matrix(c(0.021, 0.021))

#' the "dprior" function
#' Note the second parameter _must_ be called log
#' Unclear what this function does
density_function <- function(parameters, log) {
    return(0.5)
}

#' The "rprior" function that returns a matrix whose columns are the parameters
#' and each row is a sample
rnd_function <- function(num_samples) {
    return(matrix(c(3, 0.04), ncol = 2, nrow = num_samples, byrow = TRUE))
}

prior <- list("dprior" = density_function, "rprior" = rnd_function)

year_indices <- c(0L, 23L)


amis_output <- AMISforInfectiousDiseases::amis(
    prevalence_map,
    build_transmission_model(prevalence_map, fixed_parameters, year_indices, num_cores_to_use),
    prior
)

# Currently errors - I think because I
# don't know where the weights need to be set
# "No weight on any particles for locations in the active set."
print(amis_output)
