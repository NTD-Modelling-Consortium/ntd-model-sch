# Toy example: generating map samples of prevalence from a beta distribution

library("dplyr")
years <- c(2000, 2010, 2020)
num_years <- length(years)          # number of time points
M <- 200                            # number of samples per location
L <- 9                              # number of locations
location_names <- paste0("IU_",1:L)

shape1_per_location <- seq(2, len=L, by=5)

set.seed(1)
map <- as.data.frame(matrix(0.0, L*num_years, 2 + M))
colnames(map) <- c("IU_code", "year", paste0("v",1:M))
i <- 1
for(l in 1:L){
  for(j in 1:num_years){
    map_samples <- rbeta(n = M, shape1 = shape1_per_location[l], shape2 = 20)
    map[i, 3:ncol(map)] <- map_samples
    i <- i + 1
  }
}
map$IU_code <- sort(rep(location_names, num_years))
map$year <- rep(years, L)
# map[,1:6]
# dim(map)
# hist(as.numeric(map[1,-c(1,2)]))
# hist(as.numeric(map[20,-c(1,2)]))
# range((map[,-c(1,2)]))

write.csv(x = map, file = "./data/map.csv", row.names = F)

# How this data should be passed to AMIS:
map <- read.csv("./data/map.csv")

L <- 9
years <- c(2000, 2010, 2020)
num_years <- length(years)
location_names <- paste0("IU_",1:L)

prevalence_map <- vector("list", num_years)
for(j in 1:num_years){
  prev_year_j <- filter(map, year==years[j])
  prevalence_map[[j]]$data <- prev_year_j[,-c(1,2)]
}
