library(tidyverse)


india = read_csv("./data/india_processed.csv")

#'
#' we consider the following treatments:
#' - phone usage
#' - electrification
#' - paved road
#'
survey_by_state = india %>%
  group_by(state_idx) %>%
  summarise(name = first(state_name),
            n_villages = n(),
            rho_power = cor(secc_cons_per_cap_scaled,
                            pc11_vd_power_all, use = "pairwise.complete.obs"),
            mean_power = mean(pc11_vd_power_all, na.rm = TRUE),
            rho_phone = cor(secc_cons_per_cap_scaled,
                            phone_share, use = "pairwise.complete.obs"),
            mean_phone = mean(phone_share, na.rm = TRUE))  %>%
  filter(n_villages > 200)

print(survey_by_state, n=30)

ggplot(survey_by_state, aes(x = rho_phone, y = n_villages)) + geom_point()

