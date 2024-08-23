
# on cluster:
#id = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
# for testing:
id = 1
# should be "ascaris", "hookworm" or "trichuris"
species = "ascaris"
task = "projections"

set.seed(id*3)

library(dplyr)

# assuming working directory is ntd-model-sch
source("./sch_simulation/amis_integration/amis_integration.R")

args <- commandArgs(trailingOnly=TRUE)
num_cores_to_use <- parallel::detectCores()
if(length(args) == 1) {
    num_cores_to_use <- as.integer(args)
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
# load IUs and iu_task_lookup
if (species=="trichuris"){
    load("../Maps/iu_task_lookup_trichuris.rds")
    ius_list=read.csv(paste0("sch_simulation/data/endgame_inputs/IUs_MTP_projections_trichuris_",id,".csv"), header=F)
} else {
    load("../Maps/iu_task_lookup_sth.rds")
    ius_list=read.csv(paste0("sch_simulation/data/endgame_inputs/IUs_MTP_projections_",id,".csv"), header=F)
}

# run projections for each iu
for (iu in ius_list){
    fitting_id = iu_task_lookup$TaskID[which(iu_task_lookup$IU_2021 == iu)]
    
    #for testing
    seeds = 1:10
    params = cbind(R0=runif(10,1.2,2),
                   k=runif(10,0.3,0.5))
    #for real!
    #sampled_params = load(paste0("../data/endgame_inputs/sampled_params_",iu,".csv"))
    #seeds = sampled_params[,1]
    #params = sampled_params[,2:3]

    fixed_parameters <- sch_simulation$FixedParameters(
        # the higher the value of N, the more consistent the results will be
        # though the longer the simulation will take
        number_hosts = 500L,
        # no intervention
        coverage_file_name = ifelse(species=="trichuris",
                                    paste0("endgame_inputs/InputMDA_MTP_projections_trichuris_",fitting_id,".xlsx"),
                                    paste0("endgame_inputs/InputMDA_MTP_projections_",fitting_id,".xlsx")),
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
        directory = "projections", name_prefix = paste0("projections_",species,"_",iu)
    )

    transmission_model = build_transmission_model(prevalence_map, fixed_parameters, year_indices, num_cores_to_use, final_state_config = final_state_config)

    # Run projections
    projections <- transmission_model(seeds, params, n_tims)
}

en<-Sys.time()
print(as.numeric(difftime(en,st,units="mins")))