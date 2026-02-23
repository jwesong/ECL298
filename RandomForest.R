library(tidyverse)
library(randomForest)

water <- read.csv("data/total_water_use.csv")

water_data <- water[, c("TOTAL_WATER_USE", "WST_GROUNDWATER", "WST_OTHER", 
                        "WST_RECYCLED_WATER", "WST_REUSED_WATER", 
                        "WST_SURFACE_WATER", "WUS_AGRICULTURAL", 
                        "WUS_INDUSTRIAL", "WUS_MANAGED_RECHARGE",
                        "WUS_MANAGED_WETLANDS", "WUS_NATIVE_VEGETATION", 
                        "WUS_OTHER", "WUS_URBAN")]

water_data <- na.omit(water_data)


# Train/test 50/50 split
set.seed(1)
train <- sample(1:nrow(water_data), nrow(water_data) / 2)
water.test <- water_data[-train, "TOTAL_WATER_USE"]


# Bagging is Random Forest with mtry = 12 (all predictors)

set.seed(1)
bag.water <- randomForest(TOTAL_WATER_USE ~ ., 
                          data = water_data,
                          subset = train, 
                          mtry = 12,           # All 12 predictors
                          importance = TRUE)

print(bag.water)

yhat.bag <- predict(bag.water, newdata = water_data[-train, ])

# Prediction vs Actual
plot(yhat.bag, water.test,
     xlab = "Predicted Total Water Use",
     ylab = "Actual Total Water Use",
     main = "Bagging: Test Set Predictions",
     pch = 19, col = rgb(0, 0.5, 0, 0.5),
     cex.main = 1.3, font.main = 2)
abline(0, 1, col = "red", lwd = 2, lty = 2)
grid()


# Test MSE
bag_mse <- mean((yhat.bag - water.test)^2)
bag_mse

# Try different # of trees
bag.water.25 <- randomForest(TOTAL_WATER_USE ~ ., 
                             data = water_data,
                             subset = train, 
                             mtry = 12, 
                             ntree = 25)
yhat.bag.25 <- predict(bag.water.25, newdata = water_data[-train, ])
bag_mse.25 <- mean((yhat.bag.25 - water.test)^2)
bag_mse.25

# ===== RANDOM FOREST =====

set.seed(1)
rf.water <- randomForest(TOTAL_WATER_USE ~ ., 
                         data = water_data,
                         subset = train, 
                         mtry = 6,            # same as ISLR example
                         importance = TRUE)

print(rf.water)

yhat.rf <- predict(rf.water, newdata = water_data[-train, ])

# Plot predictions vs actual
plot(yhat.rf, water.test,
     xlab = "Predicted Total Water Use",
     ylab = "Actual Total Water Use",
     main = "Random Forest: Test Set Predictions",
     pch = 19, col = rgb(1, 0.5, 0, 0.5),
     cex.main = 1.3, font.main = 2)
abline(0, 1, col = "red", lwd = 2, lty = 2)
grid()


# Test MSE
rf_mse <- mean((yhat.rf - water.test)^2)
rf_mse

# =============== IMPORTANCE ===================
imp <- importance(rf.water)
print(round(imp, 2))


## Importance plot as shown in ISLR p359
varImpPlot(rf.water, 
           main = "Variable Importance Plot",
           pch = 19,
           col = "steelblue",
           cex = 1.2)


