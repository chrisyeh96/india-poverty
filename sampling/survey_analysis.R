library(tidyverse)
library(broom)
library(haven)


setwd("~/Projects/predicting-poverty")

load_survey_data = function() {

  survey_data = read_stata("data/pc0111_village_elec_road_dist_ag.dta")
  join_keys = read_csv("data/join_keys.csv")

  survey_data = survey_data %>%
    mutate(village_id = as.integer(pc11_village_id),
           electrification = as.integer(pc11_vd_power_all),
           paved_road = as.integer(pc11_vd_rd_p_btr),
           distance_to_city = as.numeric(pc11_dist_km_town_pop100),
           share_ag = as.numeric(secc_nco2d_cultiv_share)) %>%
    select(village_id, electrification, paved_road, distance_to_city, share_ag)

  print(paste("Initial survey dataset had", nrow(survey_data), "rows,",
              "and", length(unique(survey_data$village_id)), "unique ids."))

  survey_data = left_join(survey_data, join_keys,
                           by=c("village_id" = "village_id"))
  return(survey_data)
}

load_india_data = function() {

  india_data = read_csv("data/india.csv")
  india_data$daily_exp = india_data$secc_cons_per_cap_scaled / 365.25 / 16.013

  print(paste("Initial India dataset had", nrow(india_data), "rows,",
              "and", length(unique(india_data$id)), "unique ids."))
  return(india_data)
}

run_analyses = function(df) {

  cors = c(elec=cor(df$daily_exp, df$electrification),
           paved=cor(df$daily_exp, df$paved_road),
           dist=cor(df$daily_exp, df$distance_to_city),
           ag=cor(df$daily_exp, df$share_ag))

  print("-- Correlations with scaled per capita income")
  print(cors)

  coeffs_by_state = df %>%
    group_by(state_id) %>%
    summarise(rho = cor(daily_exp, electrification),
              n_village = length(daily_exp))

  plot(coeffs_by_state$rho, coeffs_by_state$n_village)
}

main = function() {

  survey_data = load_survey_data()
  india_data = load_india_data()

  join_data = left_join(india_data, survey_data, by = c("id" = "id"))

  print(paste("Sanity check: electrification has correlation of",
              cor(join_data$secc_cons_per_cap_scaled, join_data$electrification,
                  use="pairwise.complete.obs")))

  complete = join_data %>% na.omit
  print(paste("Joined dataset had", nrow(join_data), "rows."))
  print(paste("Complete dataset had", nrow(complete), "rows."))

  run_analyses(complete)

  write_csv(complete %>%
              mutate(village_id=id) %>%
              select(village_id, electrification, daily_exp),
            "data/electrification.csv")
}

main()
