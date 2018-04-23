
# Models ------------------------------------------------------------------
library(glmnet)

setwd("/Users/evanz/Dropbox/Spring 2018/Cis520/final proj/Pruned CV, Test, Train Sets")
CV1 <- read.csv("ObesityCVFold_1.csv")
CV2 <- read.csv("ObesityCVFold_2.csv")
CV3 <- read.csv("ObesityCVFold_3.csv")
CV4 <- read.csv("ObesityCVFold_4.csv")
CV5 <- read.csv("ObesityCVFold_5.csv")

Train <- read.csv("ObesityFullTrainSet.csv")
Test <- read.csv("ObesityTestSet.csv")

# OLS Linear Regression
ytest <- Test$PCT_OBESE_ADULTS13
ytrain <- Train$PCT_OBESE_ADULTS13

fit <- lm(`PCT_OBESE_ADULTS13` ~ . -1  ,data= Train)
summary(fit) # 99.54% adjusted R squared
coef(fit)
plot(residuals(fit))
MSE_Train <- mean((predict(fit,Train)- ytrain)^2) #5.74319

pred <- predict(fit, Test)
plot(pred, ytest)
title("OLS")

RMSE_Test <- sqrt(mean((pred- ytest)^2)) #6.67% Test Error

# BIC Minimizing OLS Linear Regression
library(MASS)

fitA <- lm(`PCT_OBESE_ADULTS13` ~ 0, data = Train)
AIC_glm<- MASS::stepAIC(fit, direction = 'both', scope = list(upper = fit, lower = fitA)) 

summary(AIC_glm)
MSE_Train <- mean((predict(AIC_glm,Train)- ytrain)^2)  #5.802%

pred <- predict(AIC_glm, Test)
plot(pred, ytest)

MSE_Test <- mean((pred- ytest)^2) #6.56% Test Error



# Ridge Regression - L2 ---------------------------------------------------

require(dplyr)
require(magrittr)

# Cross validation
lambda <- c(10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 10^2)
ridge_er <- matrix(, nrow= 5, ncol =8)

A <- rbind(CV2, CV3, CV4, CV5) 
B <- rbind(CV1, CV3, CV4, CV5)
C <- rbind(CV1, CV2, CV4, CV5)
D <- rbind(CV1, CV2, CV3, CV5)
E <- rbind(CV1, CV2, CV3, CV4)
cv.train.list <- list(A,B,C,D,E)

cv.test.list <- list(CV1, CV2, CV3, CV4, CV5)

for (i in 1:5){
  train <- cv.train.list[[i]]
  test <- cv.test.list[[i]]
  ytrain <- train$PCT_OBESE_ADULTS13
  ytest <- test$PCT_OBESE_ADULTS13
  
  
  train %<>% select(-c(`PCT_OBESE_ADULTS13`))
  test %<>% select(-c(`PCT_OBESE_ADULTS13`))
  
  for (j in 1:8){
    ridge <- glmnet(as.matrix(train), as.matrix(ytrain), alpha = 0, lambda = lambda[j])
    ridge.pred <- predict(ridge, s = lambda[j], newx = as.matrix(test))
    
    RMSE <- sqrt(mean((ridge.pred - ytest)^2))
    ridge_er[i,j] <- RMSE 
  }
}

cv_er_ridge = colSums(ridge_er)
bestlam <- which.min(cv_er_ridge) # Lambda = 10^-5

# On the full Dataset
ytest <- Test$PCT_OBESE_ADULTS13
ytrain <- Train$PCT_OBESE_ADULTS13

Train.mod <- Train %>% select(-c(`PCT_OBESE_ADULTS13`))
Test.mod <- Test %>% select(-c(`PCT_OBESE_ADULTS13`))

ridge <- glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 0, lambda = lambda[bestlam])
ridge.pred <- predict(ridge, s = lambda[bestlam], newx = as.matrix(Test.mod))

plot(ridge.pred, ytest)
title("Ridge Regression")

RMSE <- sqrt(mean((ridge.pred - ytest)^2))
RMSE # 2.698

# Graphic for varying lambdas vs. Linear

# LASSO Regression - L1 ---------------------------------------------------

# Cross validation
lasso_er <- matrix(, nrow= 5, ncol =8)

for (i in 1:5){
  train <- cv.train.list[[i]]
  test <- cv.test.list[[i]]
  ytrain <- train$PCT_OBESE_ADULTS13
  ytest <- test$PCT_OBESE_ADULTS13
  
  train %<>% select(-c(`PCT_OBESE_ADULTS13`))
  test %<>% select(-c(`PCT_OBESE_ADULTS13`))
  
  for (j in 1:8){
    lasso <- glmnet(as.matrix(train), as.matrix(ytrain), alpha = 1, lambda = lambda[j])
    lasso.pred <- predict(ridge, s = lambda[j], newx = as.matrix(test))
    
    RMSE <- sqrt(mean((lasso.pred - ytest)^2))
    lasso_er[i,j] <- RMSE 
  }
}

cv_er_lasso = colSums(lasso_er)
bestlam <- which.min(cv_er_lasso) # Lambda = 10^-5

# On the full Dataset
ytest <- Test$PCT_OBESE_ADULTS13
ytrain <- Train$PCT_OBESE_ADULTS13

Train.mod <- Train %>% select(-c(`PCT_OBESE_ADULTS13`))
Test.mod <- Test %>% select(-c(`PCT_OBESE_ADULTS13`))

lasso <- glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 1, lambda = lambda[bestlam])
lasso.pred <- predict(lasso, s = lambda[bestlam], newx = as.matrix(Test.mod))

plot(lasso.pred, ytest)
title("Lasso Regression")
RMSE <- sqrt(mean((lasso.pred - ytest)^2))
RMSE # 2.69705