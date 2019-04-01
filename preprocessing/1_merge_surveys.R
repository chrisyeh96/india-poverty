library(tidyverse)
library(haven)
library(progress)


india = read_csv("./data/india.csv")
aux_fields = read_stata("./data/secc_consumption_fields.dta")
survey = read_stata("./data/pc0111_village_elec_road_dist_ag.dta")
consumption = read_stata("./data/village_consumption_imputed_pc11.dta")

#'
#' 1. merge the india data with consumption data to establish pc11_village_id
#'
consumption = consumption %>%
  select(secc_hh, secc_pop, pc11_village_id) %>%
  mutate(secc_hh = as.integer(secc_hh),
         secc_pop = as.integer(secc_pop),
         pc11_village_id = as.integer(pc11_village_id))

print(paste("India:", nrow(india)))
print(paste("Consumption:", nrow(consumption)))

india$pc11_village_id = NA
j = 1
pb = progress_bar$new(total = nrow(india))
for (i in 1:nrow(india)) {
  if (india$secc_hh[i] == consumption$secc_hh[j] &
      india$secc_pop[i] == consumption$secc_pop[j]) {
    india$pc11_village_id[i] = consumption$pc11_village_id[j]
    j = j + 1
  }
  pb$tick()
}

india %>% write_csv("./data/india.csv")

#'
#' 2. merge the india data with survey data using pc11_village_id, resulting
#' in pc01_village_id as well as power, black-top road, distance to 100k+
#'
survey = survey %>%
  select(pc01_village_id, pc01_state_id,
         pc11_village_id, pc11_vd_rd_p_btr, pc11_vd_power_all,
         pc11_dist_km_town_pop100) %>%
  mutate(pc01_state_id = as.integer(pc01_state_id),
         pc01_village_id = as.integer(pc01_village_id),
         pc11_village_id = as.integer(pc11_village_id),
         pc11_vd_rd_p_btr = as.integer(pc11_vd_rd_p_btr),
         pc11_vd_power_all = as.integer(pc11_vd_power_all),
         pc11_dist_km_town_pop100 = as.numeric(pc11_dist_km_town_pop100))

india_with_survey = india %>%
  inner_join(survey, by = c("pc11_village_id" = "pc11_village_id"))

#'
#' 3. merge the survey data with auxiliary survey fields
#'
aux_fields = aux_fields %>%
  select(pc01_state_id, pc01_village_id, salary_any_share, inc_5k_plus_share,
         inc_10k_plus_share, phone_share, gt_prim_share,
         gt_prim_21_25_share) %>%
  mutate(pc01_state_id = as.integer(pc01_state_id),
         pc01_village_id = as.integer(pc01_village_id),
         salary_any_share = as.numeric(salary_any_share),
         inc_5k_plus_share = as.numeric(inc_5k_plus_share),
         inc_10k_plus_share = as.numeric(inc_10k_plus_share),
         phone_share = as.numeric(phone_share),
         gt_prim_share = as.numeric(gt_prim_share),
         gt_prim_21_25_share = as.numeric(gt_prim_21_25_share))

india_with_survey = india_with_survey %>%
  inner_join(aux_fields, by = c("pc01_village_id" = "pc01_village_id",
                                "pc01_state_id" = "pc01_state_id"))

india_with_survey %>% write_csv("./data/india_with_survey.csv")
