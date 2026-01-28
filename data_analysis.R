## Jan 28
## Data import + describe
## github setup, invite


library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)

## 1. import data (in data folder in rep)
water_data <- read_csv("data/total_water_use.csv")

'The Groundwater Sustainability Plan (GSP) Annual Report (AR) datasets contain 
the following data submitted by Groundwater Sustainability Agencies (GSA) and 
Alternative Agencies as part of their GSP AR or Alternative to GSP AR: groundwater 
extraction, surface water supply, total water use, and change in storage volumes 
for a given water year. All data was originally submitted to the Department of 
Water Resources (DWR) through the Sustainable Groundwater Management Act (SGMA) 
Portal’s AR Modules (https://sgma.water.ca.gov/portal/gspar/submitted and 
https://sgma.water.ca.gov/portal/alternative/annualreport/submitted). Data records
within each dataset correspond to either an entire basin or one of multiple GSP 
areas which collectively correspond to an entire basin.'

## 2. describe
print(names(water_data))

# structure check: data quality of panel data?
required_vars <- c("BASIN_NAME", "REPORT_YEAR")
stopifnot(all(required_vars %in% names(water_data)))

# distinct basins
n_basins <- n_distinct(water_data$BASIN_NAME)
cat("\nNumber of unique BASIN_NAMEs:", n_basins, "\n")
print(sort(unique(water_data$BASIN_NAME)))

# for each distinct basins, how many times does it appear
basin_counts <- water_data %>%
   group_by(BASIN_NAME) %>%
   summarise(
      n_records = n(),
      .groups = "drop"
   ) %>%
   arrange(desc(n_records))
print(basin_counts)


# which report_years does it appear
basin_years <- water_data %>%
   group_by(BASIN_NAME) %>%
   summarise(
      report_years = paste(sort(unique(REPORT_YEAR)), collapse = ", "),
      n_years = n_distinct(REPORT_YEAR),
      .groups = "drop"
   ) %>%
   arrange(BASIN_NAME)
print(basin_years)

# merge
basin_year_table <- water_data %>%
   count(BASIN_NAME, REPORT_YEAR) %>%
   arrange(BASIN_NAME, REPORT_YEAR)
print(basin_year_table)

# Conclusion: yes, a panel data

## Explore the variables
summary(water_data$TOTAL_WATER_USE)

# shift in water use across years, after all this year of SGMA implemented
water_data %>%
   group_by(REPORT_YEAR) %>%
   summarise(
      mean_use = mean(TOTAL_WATER_USE, na.rm = TRUE)
   ) %>%
   arrange(REPORT_YEAR)

plot(
   water_data$REPORT_YEAR,
   water_data$TOTAL_WATER_USE,
   xlab = "Year",
   ylab = "Total Water Use"
)

#  Water_use as dependent vars
water_vars <- water_data[, c(
   "TOTAL_WATER_USE",
   "WST_GROUNDWATER", "WST_OTHER", "WST_RECYCLED_WATER", 
   "WST_REUSED_WATER", "WST_SURFACE_WATER",
   "WUS_AGRICULTURAL", "WUS_INDUSTRIAL", "WUS_MANAGED_RECHARGE",
   "WUS_MANAGED_WETLANDS", "WUS_NATIVE_VEGETATION", 
   "WUS_OTHER", "WUS_URBAN"
)]

# install.packages("GGally") 
library(GGally)

wv <- water_vars
wv <- wv[, sapply(wv, is.numeric), drop = FALSE]
wv <- wv[complete.cases(wv), , drop = FALSE]

panel.scatter <- function(x, y, ...) {
   points(x, y,
          pch = 16,
          cex = 0.5,
          col = rgb(0, 0.45, 0.74, 0.35))
}

panel.label <- function(x, ...) {
   usr <- par("usr"); on.exit(par(usr))
   par(usr = c(0, 1, 0, 1))
   text(0.5, 0.5, colnames(wv)[par("mfg")["row"]], cex = 1.1, font = 2)
}

# Draw scatter plot to explore
pairs(
   wv,
   lower.panel = panel.scatter,
   upper.panel = panel.scatter,
   diag.panel  = panel.label,
   main = "Scatter matrix of water variables"
)

cor(water_data$TOTAL_WATER_USE, water_data$WUS_AGRICULTURAL)
## 0.970774, highly correltaion vairables 
