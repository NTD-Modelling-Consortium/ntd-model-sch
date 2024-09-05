
# on cluster:
id = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
# for testing:
# id = 1
# should be "ascaris", "hookworm" or "trichuris"
species = "trichuris"

set.seed(id*3)

library(dplyr)

# assuming working directory is ntd-model-sch
source("./sch_simulation/amis_integration/amis_integration.R")

args <- commandArgs(trailingOnly=TRUE)
if(length(args) == 1) {
    num_cores_to_use <- as.integer(args)
}else{
    ## for testing
    num_cores_to_use <- 1
}

print(paste0('Using ', num_cores_to_use, ' cores'))

sch_simulation <- get_amis_integration_package()

# Load prevalence map and filter rows for TaskID == id
year_indices <- c(15L,28L,33L) # 2000, 2013, 2018 (end of year in model)
load(paste0("../Maps/",species,"_maps.rds"))
prevalence_map = get(paste0(species,"_map_allyears"))
prevalence_map = lapply(1:length(prevalence_map), function(t){
  output=list(data = as.matrix(prevalence_map[[t]]$data %>% 
                                 filter(TaskID==id) %>% 
                                 select(mu, sigma)), 
              likelihood = prevalence_map[[t]]$likelihood)
  rownames(output$data) = rownames(prevalence_map[[t]]$data)[prevalence_map[[t]]$data$TaskID==id]
  return(output)
  
})
n_tims = length(prevalence_map)

st<-Sys.time()

# load IUs and proj_iu_task_lookup
load("../Maps/proj_iu_task_lookup.rds") # this loads proj_iu_task_lookup. It is for projections, and it is the same for the 3 species
ius_list <- sort(unique(proj_iu_task_lookup$IU_2021[proj_iu_task_lookup$TaskID==id]))

# run projections for each iu
for (iu in ius_list){
 
    ## for testing
    # num_seeds = 10
    # seeds = 1:num_seeds
    # params = cbind(R0=runif(num_seeds,1.2,2),
    #	 	     k=runif(num_seeds,0.3,0.5))

    ## for real!
    sampled_params = read.csv(paste0("~/NTDs/STH/post_AMIS_analysis/InputPars_MTP_",species,"/InputPars_MTP_",iu,".csv"))
    seeds = sampled_params[,"seed"]
    params = as.matrix(sampled_params[,c("R0","k")])

    fixed_parameters <- sch_simulation$FixedParameters(
        # the higher the value of N, the more consistent the results will be
        # though the longer the simulation will take
        number_hosts = 500L,
        # no intervention
        coverage_file_name = ifelse(species=="trichuris",
                                    paste0("endgame_inputs/InputMDA_MTP_projections_trichuris_",id,".xlsx"),
                                    paste0("endgame_inputs/InputMDA_MTP_projections_",id,".xlsx")),
        demography_name = "UgandaRural",
        # cset the survey type to Kato Katz with duplicate slide
        survey_type = "KK2",
        parameter_file_name = paste0("STH_params/",species,"_params_projections.txt"),
        coverage_text_file_storage_name = paste0("Man_MDA_vacc_",species,"_",iu,".txt"),
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

    final_state_config <- sch_simulation$StateSnapshotConfig(
        directory = "projections", name = paste0("projections_",species,"_",iu)
    )

    year_indices_all = min(year_indices):max(year_indices)
    transmission_model = build_transmission_model(prevalence_map, fixed_parameters, year_indices_all, num_cores_to_use, final_state_config)

    # Run projections
    all_years_result <- transmission_model(seeds, params, n_tims)
    trajectories = output
    save(trajectories, file=paste0("../trajectories/proj_trajectories_",id,"_",species,".Rdata"))

    projections <- all_years_result[,colnames(output) %in% year_indices, drop = FALSE]
}

en<-Sys.time()
print(as.numeric(difftime(en,st,units="mins")))
