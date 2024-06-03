get_venv <- function() {
    result <- system2(
        command = "pipenv",
        args = "--venv",
        stdout = TRUE,
        stderr = FALSE
    )
    if (length(result) == 0) {
        # See the main README under How to run for instructions
        # on how to setup the virtual environment
        stop("No Python virtual enviroment found,
      install amis_intergration before running this script
      and ensure the working directory is inside the project")
    }
    return(result)
}
