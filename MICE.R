# CIS520 Final Project: Social Determinants of Obesity

# Imputation of Missing (NAs) data
# Set wd and load packages

setwd("/Users/evanz/Dropbox/Spring 2018/Cis520/final proj")
library(mice)
library(dplyr)
library(magrittr)
library(VIM)
library(missForest)

# load the data
df = read.csv("Cleaned 4_16.csv")
#df <- read.csv("Cleaned noncor.csv")

utils::View(df)
summary(df)

# Check % missing data
pMiss <- function(x) {sum(is.na(x))/length(x)*100}
miss <- apply(df, 2, pMiss)

# Remove all columns with missing data above a threshold
threshold <- c(5,10,15,20,25,30) # in %

for (j in 1:length(threshold)){
  remove_list <- vector();
  for (i in 1:length(miss)){
    if (miss[i] > threshold[j]) {
      remove_list <- c(remove_list, names(miss[i]))
    }
  }
  print(length(remove_list))
}

# select threshold at 15% NA's. Removes 44 col's from full dataset.

threshold <- 5
remove_list <- vector();

  for (i in 1:length(miss)){
    if (miss[i] > threshold) {
      remove_list <- c(remove_list, names(miss[i]))
    }
  }
remove_list


#df_t <- df %>% subset(., select= -c(remove_list, "x"))
df_t <- df[, -which(names(df) %in% remove_list)]
df_t <- df_t[, -c(1)]
df_t <- sapply(df_t, as.numeric)
ncol(df_t) # 178 variables left, 

# Visualize the data
md.pattern(df_t)
df_aggr = aggr(df_t, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                   labels=names(df_t), cex.axis=.7, gap=3, 
                   ylab=c("Proportion of missingness","Missingness Pattern"))


# Analyze correlation of variables

corr_df <- df_t[1:150]
corr_df <- na.omit(corr_df)
cortotal <- cor(corr_df)

plot.new()
colnames(cortotal) <- seq(1, length(colnames(cortotal)))
rownames(cortotal) <- seq(1, length(colnames(cortotal)))
corrplot(cortotal, method = "color", mar = c(1,1,1,1))

# Remove variables with high correlations 
nums <- c(1:25,158:184)
df_s <- df_t[, -nums]

corr_df <- df_s
corr_df <- na.omit(corr_df)
cortotal <- cor(corr_df)

plot.new()
colnames(cortotal) <- seq(1, length(colnames(cortotal)))
rownames(cortotal) <- seq(1, length(colnames(cortotal)))
corrplot(cortotal, method = "color", mar = c(1,1,1,1))
# Imputation --------------------------------------------------------------


# Start data imputation
# Try easy mean imputation
imp <- mice(df_t, method = 'mean', printFlag = FALSE, m=1, maxit = 1)
df_i <- complete(imp)

fit <- with(imp, lm(`PCT_OBESE_ADULTS13` ~ ., data = df_i))
summary(fit)

write.csv(df_i, file= 'MeanImp.csv', row.names = FALSE)

#random Forest imputation
missForest(df_t, maxiter= 5, ntree = 5)

# CART imputation
imp2 <- mice(df_t, method = 'cart', printFlag = FALSE, m=1, maxit = 5) #178 variables
df_i2 <- complete(imp2)

fit <- with(imp2, lm(`PCT_OBESE_ADULTS13` ~. ,data= df_i2))
summary(fit)

write.csv(df_i2, file = 'CartImp.csv')




