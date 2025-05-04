
set.seed(123)

library(actuar)  # Pakki til að herma pareto
x <- rpareto(1000, 2, 2)  # Búa til Dummy gögn

library(evd) # Fyrir Generalized Pareto Distribution - Notað til að finna F_u
library(tidyverse)

# Eftirfarandi kóði fittar "F_small og F_large"
# Þetta er í raun kóðinn í Total Claim Amount á bls 20
q <- 0.95  # Vel einhvern quantile
threshold <- quantile(x, q) 


# Separate small and large claims
small_claims_data <- x[x <= threshold]
large_claims_data <- x[x > threshold]
# Estimate small claims distribution (F_s) using empirical CDF
F_s <- ecdf(small_claims_data) # Note this is F_{u-}
# Estimate large claims distribution (F_u) using Generalized Pareto Distribution (GPD)
gpd_fit <- fpot(large_claims_data, threshold = threshold)
F_u <- function(x) {
  ifelse(x > threshold,
         pgpd(x -  threshold ,
              loc = 0,
              scale = gpd_fit$estimate[1],
              shape = gpd_fit$estimate[2]),
         0)
}
# Combine F(x) = qF_s(x) + (1-q)F_u(x)
F <- function(x) {
  q * F_s(x) + (1 - q) * F_u(x)
}


# Skoðum hvort þetta líti ekki ok út
# Create a sequence of x values for plotting
x_values <- seq(min(x), max(x), length.out = 1000)
# Compute F(x) values
F_values <- sapply(x_values, F)
# Prepare empirical CDF for the entire dataset
empirical_cdf <- ecdf(x)
empirical_values <- sapply(x_values, empirical_cdf)
# Prepare data for plotting
plot_data <- data.frame(
  x = x_values,
  F = F_values,
  Empirical = empirical_values
)


# Plot the estimated distribution and the empirical distribution
ggplot(plot_data, aes(x = x)) +
  geom_line(aes(y = F), color = "darkblue", size = 1, linetype = "solid") +
  geom_line(aes(y = Empirical), color = "darkred", size = 1, linetype = "dashed") +
  
  labs(
    title = "Estimated Distribution vs Empirical Distribution",
    x = "x",
    y = "F(x)"
  ) +
  theme_minimal() +
  ylim(c(0.9, 1))

## Lookar ok


# Við erum komin með F. Hvernig getum við hermt X?

simulate_X <- function(u, q){
  # u er uniform
  # q er threshold (samsvarar threhold að ofan)
  
  if(u < q){
    # herma úr smá tjónum, hér geri ég ráð fyrir
    # að smá tjón eru bara empirical dreifingin
    x <- sample(small_claims_data,size = 1)  
  }else{
    # Herma úr excess, nota stikana sem
    # voru fundinr með fallinu gpd_fit
    x <- rgpd(1,
              loc = 0,
              scale = gpd_fit$estimate[1],
              shape = gpd_fit$estimate[2])
    
    x <-  x + threshold
  }
  
  return(unname(x))
  
}
# Hermi eitt stak
simulate_X(runif(1), q)


# Til að herma mörg: vectorize-a fallið. Líka hægt að keyra fallið að ofan í for loop-u
simulate_X <- Vectorize(simulate_X, vectorize.args = "u")
X_simulated <- simulate_X(runif(10000), q)






# Að auki er hægt að herma úr hvaða F_X með uniform og
# helmingunaraðferð Svipað og þegar við hermdum komutíma T fyrir missleit Poisson

# Finnur x fyrir g(x) = y
# hjá okkur er y = u uniform
# og g = F dreififallið
# Passa að interval sé nógu stórt
inverse_f <- function(y, g) {
  uniroot(function(x) g(x) - y, interval = c(0, 1000))$root
}
# Viljum aftur geta keyrt fallið á vigur af gildum
inverse_f <- Vectorize(inverse_f, vectorize.args = "y")

x_simulated_numerical_inverse <- inverse_f(runif(10000), F)

# berum saman gögnin á log skala


ggplot() + 
  geom_density(aes(x = log(X_simulated), fill = "simulation 1", alpha = 0.4))+
  geom_density(aes(x = log(x_simulated_numerical_inverse), , fill = "simulation 2", alpha = 0.4))+
  theme_bw(base_size = 24)

# Mjög svipað




