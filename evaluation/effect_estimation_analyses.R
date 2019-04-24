library(tidyverse)
library(ggthemes)
library(rsample)
library(latex2exp)


r2_score = function(y, y_hat) {
  1 - sum((y_hat - y) ^ 2) / sum((y - mean(y)) ^ 2)
}

df = read_csv("./results/fold_india/effect_estimation_df.csv")
df = df %>% group_by(state_idx) %>% filter(n() > 150) %>% ungroup()

df %>% group_by(state_name, state_idx) %>% count() %>% print(n = 25)

# ==
# exploratory analysis of electrification by state
initial_results = df %>% group_by(state_name) %>%
  summarise(true_cor = cor(true, electrification),
            pred_cor = cor(pred, electrification),
            r2_score = r2_score(true, pred),
            state_idx = first(state_idx),
            cons_mean = mean(true),
            cons_sd = sd(true),
            elec_prevalence = mean(electrification),
            sat_cor = cor(true, pred),
            n_villages = n()) %>% na.omit

initial_results %>%
  ggplot(aes(x = elec_prevalence, y = true_cor)) + geom_point() +
  theme_bw() + geom_hline(yintercept = 0, lty = 2) +
  geom_vline(xintercept = 0.9, lty = 2) +
  geom_vline(xintercept = 0.1, lty = 2) +
  labs(x = "Prevalence of electricity", y = TeX("True correlation"))

relevant_results = initial_results %>%
  filter(elec_prevalence < 0.9 & elec_prevalence > 0.1)

relevant_results %>%
  ggplot(aes(x = true_cor, y = pred_cor, color = r2_score)) +
  geom_point() + geom_abline(slope = 1, intercept = 0, lty = 2) +
  ylim(0, 0.5) + xlim(0, 0.5) +
  geom_hline(yintercept = 0, lty = 2) + geom_vline(xintercept = 0, lty = 2) +
  theme_bw() + scale_color_gradient2_tableau() +
  labs(x = TeX("True $\\beta$"), y = TeX("Pred $\\beta$"),
       color = TeX("Sat $R^2$"))


