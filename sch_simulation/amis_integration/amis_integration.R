get_amis_integration_package <- function() {
  importlib <- reticulate::import("importlib")
  sch_simulation <- reticulate::import("sch_simulation.amis_integration.amis_integration")
  importlib$reload(sch_simulation)
  return(sch_simulation)
}

build_transmission_model <- function(prevalence_map, fixed_parameters, year_indices) {
  sch_simulation <- get_amis_integration_package()
  transmission_model <- function(seeds, params, n_tims = 2) {
    output <- sch_simulation$run_model_with_parameters(
      seeds, params, fixed_parameters, as.array(year_indices)
    )
    print("Output:")
    print(output)
    return(output)
  }

  return(transmission_model)
}
