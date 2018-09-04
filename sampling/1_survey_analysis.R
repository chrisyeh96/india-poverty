library(tidyverse)
library(haven)


main = function() {

  survey_data = read_stata("../data/pc0111_village_elec_road_dist_ag.dta")

  survey_data = survey_data %>%
    mutate(village_id = as.integer(pc11_village_id),
           electrification = as.integer(pc11_vd_power_all),
           paved_road = as.integer(pc11_vd_rd_p_btr),
           distance_to_city = as.numeric(pc11_dist_km_town_pop100),
           share_ag = as.numeric(secc_nco2d_cultiv_share)) %>%
    select(village_id, electrification, paved_road, distance_to_city, share_ag)

  print(paste("Initial survey dataset had", nrow(survey_data), "rows,",
              "and", length(unique(survey_data$village_id)), "unique ids."))

  india_data = read_csv("../data/india.csv")
  print(paste("Initial survey dataset had", nrow(india_data), "rows,",
              "and", length(unique(india_data$id)), "unique ids."))

  join_data = inner_join(survey_data, india_data, by = c("village_id" = "id"))
  complete = join_data %>% na.omit

  print(paste("Joined dataset had", nrow(join_data), "rows."))
  print(paste("Complete dataset had", nrow(complete), "rows."))

  model = lm(electrification ~ secc_cons_per_cap_scaled, complete)

  cors = c(elec=cor(complete$secc_cons_per_cap_scaled, complete$electrification),
           paved=cor(complete$secc_cons_per_cap_scaled, complete$paved_road),
           dist=cor(complete$secc_cons_per_cap_scaled, complete$distance_to_city),
           ag=cor(complete$secc_cons_per_cap_scaled, complete$share_ag))

  print("-- Correlations with scaled per capita income")
  print(cors)
}

main()
