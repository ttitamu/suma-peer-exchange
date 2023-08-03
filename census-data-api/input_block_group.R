library(sf)
library(mapview)
library(dplyr)
library(lubridate)
library(raster)
library(censusapi)

# Author: Gargi Singh, Assistant Research Scientist, TTI
# Date: 2020-07-01
# Description: This script downloads decennial 2010 data from US census website using API

readRenviron("C:/Users/G-Singh/Documents/.Renviron")
# Add key to .Renviron
# Sys.unsetenv(census_blockgroup_KEY)
Sys.setenv(census_KEY='TYPE YOUR CENSUS API KEY HERE')
# Reload .Renviron
readRenviron("~/.Renviron")
# Check to see that the expected key is output in your R console
Sys.getenv("census_KEY")


apis <- listCensusApis()
View(apis)

vars <- listCensusMetadata(name = "2010/dec/sf1", 
                           type = "variables")

geography <- listCensusMetadata(name = "2010/dec/sf1", 
                                type = "geography") #Tract is the smallest geography

##------------------------------- POPULATION  -------------------------------##
census_blockgroup_pop <- getCensus(name = "dec/sf1",
                              vintage = 2010,
                              vars = c("P001001", "P043001"), 
                              region = "block group:*",
                              regionin = "state:48+county:469"
)
colnames(census_blockgroup_pop)[ncol(census_blockgroup_pop)] <- "grp_qtr_pop"
colnames(census_blockgroup_pop)[ncol(census_blockgroup_pop)-1] <- "total_pop"
census_blockgroup_pop$pop_wo_sgz = census_blockgroup_pop$total_pop-census_blockgroup_pop$grp_qtr_pop
census_blockgroup_pop$GEOID10 <- as.double(paste0(census_blockgroup_pop$state, census_blockgroup_pop$county, census_blockgroup_pop$tract, census_blockgroup_pop$block_group))

write.csv(census_blockgroup_pop, "model_inputs/block_group/census_blockgroup_population.csv", row.names = F)

##------------------------------- RACE  -------------------------------##
census_blockgroup_race <- getCensus(name = "dec/sf1",
                               vintage = 2010,
                               vars = c("P001001", "P004003", "P005003", "P005004", "P005005", "P005006", "P005007", "P005008", "P005009"), 
                               region = "block group:*",
                               regionin = "state:48+county:469")
colnames(census_blockgroup_race)[5:ncol(census_blockgroup_race)] <- c("total_race", "hispanic_latino", "white", "black_african_american", "american_indian_alaska_native", "asian", 
                                                            "native_hawaiian_pacific_islander", "other_race", "two_or_more_race")
census_blockgroup_race <- census_blockgroup_race %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_race, "model_inputs/block_group/census_blockgroup_race.csv", row.names = F)

##------------------------------- GENDER  -------------------------------##
census_blockgroup_gender <- getCensus(name = "dec/sf1",
                                 vintage = 2010,
                                 vars = c("P012001","P012002", "P012026"), 
                                 region = "block group:*",
                                 regionin = "state:48+county:469")
colnames(census_blockgroup_gender)[5:ncol(census_blockgroup_gender)] <- c("total_gender", "male", "female")
census_blockgroup_gender <- census_blockgroup_gender %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_gender, "model_inputs/block_group/census_blockgroup_gender.csv", row.names = F)


##------------------------------- AGE - MALE  -------------------------------##
census_blockgroup_age_male <- getCensus(name = "dec/sf1",
                                   vintage = 2010,
                                   vars = c("P012002","P012003", "P012004", "P012005","P012006", "P012007","P012008","P012009", "P012010",
                                            "P012011", "P012012", "P012013","P012014","P012015","P012016","P012017","P012018","P012019",
                                            "P012020", "P012021","P012022","P012023","P012024","P012025"), 
                                   region = "block group:*",
                                   regionin = "state:48+county:469")
colnames(census_blockgroup_age_male)[5:ncol(census_blockgroup_age_male)] <- c("total_male", "male_under5","male_5_9", "male_10_14", "male_15_17", "male_18_19", "male_20", "male_21", "male_22_24", 
                                                                    "male_25_29", "male_30_34", "male_35_39", "male_40_44", "male_45_49", "male_50_54", "male_55_59", "male_60_61", "male_62_64", 
                                                                    "male_65_66", "male_67_69", "male_70_74", "male_75_79", "male_80_84", "male_85_over")
census_blockgroup_age_male <- census_blockgroup_age_male %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_age_male, "model_inputs/block_group/census_blockgroup_age_male.csv", row.names = F)

##------------------------------- AGE - FEMALE  -------------------------------##
census_blockgroup_age_female <- getCensus(name = "dec/sf1",
                                     vintage = 2010,
                                     vars = c("P012026", "P012027","P012028","P012029", "P012030","P012031","P012032","P012033","P012034","P012035",
                                              "P012036", "P012037","P012038", "P012039","P012040","P012041","P012042","P012043","P012044","P012045",
                                              "P012046", "P012047","P012048","P012049"), 
                                     region = "block group:*",
                                     regionin = "state:48+county:469")
colnames(census_blockgroup_age_female)[5:ncol(census_blockgroup_age_female)] <- c("total_female", "female_under5","female_5_9", "female_10_14", "female_15_17", "female_18_19", "female_20", "female_21", "female_22_24", 
                                                                        "female_25_29", "female_30_34", "female_35_39", "female_40_44", "female_45_49", "female_50_54", "female_55_59", "female_60_61", "female_62_64", 
                                                                        "female_65_66", "female_67_69", "female_70_74", "female_75_79", "female_80_84", "female_85_over")
census_blockgroup_age_female <- census_blockgroup_age_female %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_age_female, "model_inputs/block_group/census_blockgroup_age_female.csv", row.names = F)

##--------------------------------- MEDIAN AGE  ---------------------------------##
census_blockgroup_median_age <- getCensus(name = "dec/sf1",
                                     vintage = 2010,
                                     vars = c("P013001","P013002","P013003"), 
                                     region = "block group:*",
                                     regionin = "state:48+county:469")
colnames(census_blockgroup_median_age)[5:ncol(census_blockgroup_median_age)] <- c("median_age", "male_median_age","female_median_age")
census_blockgroup_median_age <- census_blockgroup_median_age %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_median_age, "model_inputs/block_group/census_blockgroup_median_age.csv", row.names = F)

##--------------------------------- HOUSEHOLD TYPE  ---------------------------------##
census_blockgroup_hh_type <- getCensus(name = "dec/sf1",
                                  vintage = 2010,
                                  vars = c("P018001","P018002","P018007"), 
                                  region = "block group:*",
                                  regionin = "state:48+county:469")
colnames(census_blockgroup_hh_type)[5:ncol(census_blockgroup_hh_type)] <- c("total_family", "family","non_family")
census_blockgroup_hh_type <- census_blockgroup_hh_type %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_hh_type, "model_inputs/block_group/census_blockgroup_hh_type.csv", row.names = F)

##--------------------------------- CHILDREN  ---------------------------------##
census_blockgroup_children_under18 <- getCensus(name = "dec/sf1",
                                           vintage = 2010,
                                           vars = c("P039003", "P039004", "P039006"), 
                                           region = "block group:*",
                                           regionin = "state:48+county:469")
colnames(census_blockgroup_children_under18)[5:ncol(census_blockgroup_children_under18)] <- c("family_with_children", "family_with_under_6","family_with_6to17")
census_blockgroup_children_under18 <- census_blockgroup_children_under18 %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_children_under18, "model_inputs/block_group/census_blockgroup_children_under18.csv", row.names = F)

##--------------------------------- AVERAGE HH SIZE  ---------------------------------##
census_blockgroup_hh_size <- getCensus(name = "dec/sf1",
                                  vintage = 2010,
                                  vars = c("P017001"), 
                                  region = "block group:*",
                                  regionin = "state:48+county:469")
colnames(census_blockgroup_hh_size)[5:ncol(census_blockgroup_hh_size)] <- c("hh_size")
census_blockgroup_hh_size <- census_blockgroup_hh_size %>% left_join(census_blockgroup_pop[,c("state", "county", "tract", "block_group", "GEOID10")])

write.csv(census_blockgroup_hh_size, "model_inputs/block_group/census_blockgroup_hh_size.csv", row.names = F)


