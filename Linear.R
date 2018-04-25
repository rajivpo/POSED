
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
system.time({lm(`PCT_OBESE_ADULTS13` ~ . -1  ,data= Train)})
summary(fit) # 99.54% adjusted R squared
coef(fit)
plot(residuals(fit))
MSE_Train <- mean((predict(fit,Train)- ytrain)^2) #5.74319
RMSE_Train <- sqrt(MSE_Train) # 2.396495

pred <- predict(fit, Test)
plot(pred, ytest)
title("OLS")

RMSE_Test <- sqrt(mean((pred- ytest)^2)) #2.58% Test Error

#write.csv(as.data.frame(coef(fit)), file = 'OLS.csv')

# Standardized OLS
Train_a <- scale(Train, center = TRUE, scale = TRUE)
Test_a <-  scale(Test, center = TRUE, scale = TRUE)

fit_a <- lm(`PCT_OBESE_ADULTS13` ~ . -1  ,data= Train)
summary(fit_a) # 99.54% adjusted R squared
coef(fit_a)
plot(residuals(fit_a))
MSE_Train <- mean((predict(fit_a,Train)- ytrain)^2) #5.74319

pred <- predict(fit_a, Test)
plot(pred, ytest)
title("OLS_standardized")

RMSE_a <- sqrt(mean((pred- ytest)^2)) #2.58% Test Error
RMSE_a #2.583%

#write.csv(as.data.frame(coef(fit_a)), file = 'OLS_stand.csv')


# AIC Minimizing OLS ------------------------------------------------------
library(MASS)
ytest <- Test$PCT_OBESE_ADULTS13
ytrain <- Train$PCT_OBESE_ADULTS13

# From the Top
fitA <- lm(`PCT_OBESE_ADULTS13` ~ 0, data = as.data.frame(Train))
AIC_glm<- MASS::stepAIC(fit, direction = 'both', scope = list(upper = fit, lower = fitA)) 
system.time ({MASS::stepAIC(fit, direction = 'both', scope = list(upper = fit, lower = fitA))}) # Timing

summary(AIC_glm)
MSE_Train <- mean((predict(AIC_glm,as.data.frame(Train))- ytrain)^2)  #5.802%
RMSE_Train <- sqrt(MSE_Train)

pred <- predict(AIC_glm, as.data.frame(Test))
plot(pred, ytest)

MSE_Test <- mean((pred- ytest)^2) #6.56
RMSE_Test <- sqrt(MSE_Test) #2.56
#write.csv(as.data.frame(as.matrix(AIC_glm$coefficients)), file = 'OLS_AIC.csv')

keep <- names(AIC_glm$model)[-1]
index <- match(keep, colnames(Train)) # Indices of included variables
AIC_coefs <- AIC_glm$coefficients

all_coef <- vector(mode = 'numeric', length = ncol(Train))
j <- 1
for (i in 1:length(all_coef)){
  if (i %in% index){
    all_coef[i] <- AIC_coefs[j]; 
    j <- j + 1;
  }
  else
    all_coef[i] <- 0;
}

write.csv(as.data.frame(as.matrix(all_coef)), file = 'OLS_AIC.csv')


# From the bottom

AIC_glm2 <- MASS::stepAIC(fitA, direction = 'both', scope = list(upper = fit, lower = fitA)) 
system.time ({MASS::stepAIC(fitA, direction = 'both', scope = list(upper = fit, lower = fitA))}) # Timing

summary(AIC_glm2)
MSE_Train <- mean((predict(AIC_glm2,as.data.frame(Train))- ytrain)^2)  #5.802%
RMSE_Train <- sqrt(MSE_Train)
RMSE_Train #2.653865

pred <- predict(AIC_glm2, as.data.frame(Test))
plot(pred, ytest)

MSE_Test <- mean((pred- ytest)^2) #8.267064
RMSE_Test <- sqrt(MSE_Test) 
RMSE_Test #2.87525

# Ridge Regression - L2 ---------------------------------------------------

require(dplyr)
require(magrittr)
require(glmnet)

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
  ytrain <- scale(ytrain, center = TRUE, scale = TRUE)
  ytest <- test$PCT_OBESE_ADULTS13
  ytest <- scale(ytest, center = TRUE, scale = TRUE)
  
  
  train %<>% select(-c(`PCT_OBESE_ADULTS13`))
  train <- scale(train, center = TRUE, scale = TRUE)
  test %<>% select(-c(`PCT_OBESE_ADULTS13`))
  test <- scale(test, center = TRUE, scale = TRUE)
  
  
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
mu <- mean(ytest)
sd <- sqrt(var(ytest))
ytest <- scale(ytest, center = TRUE, scale = TRUE)

ytrain <- Train$PCT_OBESE_ADULTS13
mu_train <- mean(ytrain)
sd_train <- sqrt(var(ytrain))
ytrain <- scale(ytrain, center = TRUE, scale = TRUE)


Train.mod <- Train %>% select(-c(`PCT_OBESE_ADULTS13`))
Train.mod <- scale(Train.mod, center = TRUE, scale = TRUE)
Test.mod <- Test %>% select(-c(`PCT_OBESE_ADULTS13`))
Test.mod <- scale(Test.mod, center= TRUE, scale = TRUE)

ridge <- glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 0, lambda = lambda[bestlam])
system.time({glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 0, lambda = lambda[bestlam])})
ridge.pred <- predict(ridge, s = lambda[bestlam], newx = as.matrix(Test.mod))
ridge.predtrain <- predict(ridge, s = lambda[bestlam], newx = as.matrix(Train.mod))

plot(ridge.pred, ytest)
title("Ridge Regression")

ridge.pred_un <- ridge.pred * sd + mu
ytest_un <- ytest * sd + mu
RMSE_test <- sqrt(mean((ridge.pred_un - ytest_un)^2))
RMSE_test # 2.698 (unstandardized), 2.741105 (standardized)

ridge.pred_unt <- ridge.predtrain * sd_train + mu_train
ytrain_un <- ytrain * sd_train + mu_train
RMSE_train <- sqrt(mean((ridge.pred_unt - ytrain_un)^2))
RMSE_train # 2.479725

coef <- ridge$beta 
#write.csv(as.data.frame(as.matrix(coef)), file = 'Ridge_coef.csv')

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
  train <- scale(train, center = TRUE, scale = TRUE)
  test %<>% select(-c(`PCT_OBESE_ADULTS13`))
  test <- scale(test, center = TRUE, scale = TRUE)
  
  for (j in 1:8){
    lasso <- glmnet(as.matrix(train), as.matrix(ytrain), alpha = 1, lambda = lambda[j])
    lasso.pred <- predict(lasso, s = lambda[j], newx = as.matrix(test))
    
    RMSE <- sqrt(mean((lasso.pred - ytest)^2))
    lasso_er[i,j] <- RMSE 
  }
}

cv_er_lasso = colSums(lasso_er)/5
bestlam <- which.min(cv_er_lasso) # Lambda = 10^-4

# On the full Dataset
ytest <- Test$PCT_OBESE_ADULTS13
mu <- mean(ytest)
sd <- sqrt(var(ytest))
ytest <- scale(ytest, center = TRUE, scale = TRUE)

ytrain <- Train$PCT_OBESE_ADULTS13
mu_train <- mean(ytrain)
sd_train <- sqrt(var(ytrain))
ytrain <- scale(ytrain, center = TRUE, scale = TRUE)


Train.mod <- Train %>% select(-c(`PCT_OBESE_ADULTS13`))
Train.mod <- scale(Train.mod, center = TRUE, scale = TRUE)
Test.mod <- Test %>% select(-c(`PCT_OBESE_ADULTS13`))
Test.mod <- scale(Test.mod,  center = TRUE, scale = TRUE)

lasso <- glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 1, lambda = lambda[bestlam])
system.time({glmnet(as.matrix(Train.mod), as.matrix(ytrain), alpha = 1, lambda = lambda[bestlam])})
lasso.pred <- predict(lasso, s = lambda[bestlam], newx = as.matrix(Test.mod))
lasso.predtrain <-predict(lasso, s = lambda[bestlam], newx = as.matrix(Train.mod))

plot(lasso.pred, ytest)
title("Lasso Regression")

lasso.pred_un <- lasso.pred * sd + mu
ytest_un <- ytest * sd + mu
RMSE_test <- sqrt(mean((lasso.pred_un - ytest_un)^2))
RMSE_test # 2.739308(standardized)

lasso.pred_unt <- lasso.predtrain * sd_train + mu_train
ytrain_un <- ytrain * sd_train + mu_train
RMSE_train <- sqrt(mean((lasso.pred_unt - ytrain_un)^2))
RMSE_train # 2.477401

coef2 <- lasso$beta 
#write.csv(as.data.frame(as.matrix(coef2)), file = 'Lasso_coef.csv')


# Elastic Net -------------------------------------------------------------


