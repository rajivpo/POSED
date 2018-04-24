setwd("/Users/joshuaogunleye/Desktop/CIS 520/Project/Obesity CV, Train, Test Data")
require(data.table)
require(rpart)
install.packages("rpart.plot")
require(rpart.plot)


## CV Folds
cvTestF1 <- read.csv("ObesityCVFold_1.csv")
cvTestF2 <- read.csv("ObesityCVFold_2.csv")
cvTestF3 <- read.csv("ObesityCVFold_3.csv")
cvTestF4 <- read.csv("ObesityCVFold_4.csv")
cvTestF5 <- read.csv("ObesityCVFold_5.csv")


cvTrainFolds1 = rbind(cvTestF2,cvTestF3,cvTestF4,cvTestF5)
cvTrainFolds2 = rbind(cvTestF1,cvTestF3,cvTestF4,cvTestF5)
cvTrainFolds3 = rbind(cvTestF1,cvTestF2,cvTestF4,cvTestF5)
cvTrainFolds4 = rbind(cvTestF1,cvTestF2,cvTestF3,cvTestF5)
cvTrainFolds5 = rbind(cvTestF1,cvTestF2,cvTestF3,cvTestF4)

## TO-DO: Seperate Labels in for-loop
cvTestList = list(cvTestF1,cvTestF2,cvTestF3,cvTestF4,cvTestF5)
cvTrainList = list(cvTrainFolds1,cvTrainFolds2,cvTrainFolds3,cvTrainFolds4,cvTrainFolds5)


## Seperate Labels From Data
fullTestData = read.csv("ObesityTestSet.csv")
testLabels = fullTestData[,which(names(fullTestData)=="PCT_OBESE_ADULTS13")] #get labels
testData = fullTestData[,-which(names(fullTestData)=="PCT_OBESE_ADULTS13")]

fullTrainData = read.csv("ObesityFullTrainSet.csv")
trainLabels = fullTrainData[,which(names(fullTrainData)=="PCT_OBESE_ADULTS13")] #get labels
trainData = fullTrainData[,-which(names(fullTrainData)=="PCT_OBESE_ADULTS13")]



########################################################################

#rpart Package 

########################################################################

#fit inital model to training data
fit1 = rpart(trainLabels ~ .,data = trainData, method = "anova")
pfit1 = prune(fit1, cp = fit1$cptable[which.min(fit1$cptable[,"xerror"]),"CP"])
summary(pfit1)
plot(pfit1)
text(pfit1)



## CV Folds on Min Sample Per Split (Choose 30)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
minSamples = c(1,5,15,30,60)

for (i in 1:5) {
  ctrl = rpart.control(minsplit = minSamples[i], cp = 0)
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = rpart(currLabels ~., data = currData, method = "anova", control = ctrl)
    tempFitPrune = prune(tempFit, cp = tempFit$cptable[which.min(tempFit$cptable[,"xerror"]),"CP"])
    
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFitPrune, newdata = currTestData, type = "vector")
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
    
  }
  
  fullTrainFit = rpart(trainLabels~., data = trainData, control = ctrl)
  fullTrainFitPruned = prune(fullTrainFit, cp = fullTrainFit$cptable[which.min(fullTrainFit$cptable[,"xerror"]),"CP"])
  trainPred = predict(fullTrainFitPruned, newdata = trainData, type = "vector")
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFitPruned, newdata = testData, type = "vector")
  testMSE[i] = mse(testLabels, testPred)
  
}
## cvMSE =  20.285890  9.255170  9.209743  8.999613  9.316311
## trainMSE = 20.283671  5.896577  5.523530  6.598584  6.945326
## testMSE = 21.177001  9.285521  9.159120  9.140059  8.861380


## CV Folds on Max Depth of Tree (Choose 30)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
maxDepth = c(3,7,15,24,30)

for (i in 1:5) {
  ctrl = rpart.control(maxdepth = maxDepth[i], cp = 0)
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = rpart(currLabels ~., data = currData, method = "anova", control = ctrl)
    tempFitPrune = prune(tempFit, cp = tempFit$cptable[which.min(tempFit$cptable[,"xerror"]),"CP"])
    
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFitPrune, newdata = currTestData, type = "vector")
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
  }
  fullTrainFit = rpart(trainLabels~., data = trainData, control = ctrl)
  fullTrainFitPruned = prune(fullTrainFit, cp = fullTrainFit$cptable[which.min(fullTrainFit$cptable[,"xerror"]),"CP"])
  trainPred = predict(fullTrainFitPruned, newdata = trainData, type = "vector")
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFitPruned, newdata = testData, type = "vector")
  testMSE[i] = mse(testLabels, testPred)
  
}

## cvMSE =  11.991467  9.130883  9.270173  8.994601  9.113388
## trainMSE = 11.096172  6.349414  6.060087  5.317765  5.566926
## testMSE =  11.928113  9.109271  9.232293  9.060326  9.111288


## CV Folds on Min # of observations per leaf node (Choose 7)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
minBucket = c(1,3,7,15,30)

for (i in 1:5) {
  ctrl = rpart.control(minbucket = minBucket[i], cp = 0)
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = rpart(currLabels ~., data = currData, method = "anova", control = ctrl)
    tempFitPrune = prune(tempFit, cp = tempFit$cptable[which.min(tempFit$cptable[,"xerror"]),"CP"])
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFitPrune, newdata = currTestData, type = "vector")
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
    
  }
  fullTrainFit = rpart(trainLabels~., data = trainData, control = ctrl)
  fullTrainFitPruned = prune(fullTrainFit, cp = fullTrainFit$cptable[which.min(fullTrainFit$cptable[,"xerror"]),"CP"])
  trainPred = predict(fullTrainFitPruned, newdata = trainData, type = "vector")
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFitPruned, newdata = testData, type = "vector")
  testMSE[i] = mse(testLabels, testPred)
}

## cvMSE =  9.109350 9.121641 8.994115 9.264786 9.457640
## trainMSE = 7.106331 6.197111 5.864995 5.944226 6.688464
## testMSE = 9.007015 9.324055 9.186933 8.740349 8.847440


## RPART Final Results
 
ctrl = rpart.control(minsplit =30,minbucket = 7, maxdepth = 30, cp = 0)
finalFit = rpart(trainLabels ~ .,data = trainData, method = "anova", control = ctrl)
pFinalFit = prune(finalFit, cp = finalFit$cptable[which.min(finalFit$cptable[,"xerror"]),"CP"])

finalTrainPred = predict(pFinalFit,trainData, type = "vector")
trainMSE = mse(trainLabels, trainPred)

finalTestPred = predict(pFinalFit,testData, type = "vector")
testMSE = mse(testLabels, testPred)

##trainMSE = 6.688464
##testMSE = 8.84744

prp(pFinalFit,varlen=-5, digits = 3,split.cex = 3)
text(pFinalFit)

########################################################################

#Partykit Package (Conditional Inference/Significance Tests to Choose Splits)

########################################################################
install.packages("partykit")
require(partykit)


#fit inital model to training data
partyfit = ctree(trainLabels~.,data = trainData)
print(partyfit)
plot(partyfit)


## CV Folds on Min Sample Per Split (Choose 5)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
minSamples = c(1,5,15,30,60)

for (i in 1:5) {
  ctrl = ctree_control(minsplit = minSamples[i])
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = ctree(currLabels ~., data = currData, method = "anova", control = ctrl)
    
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFit, newdata = currTestData)
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
    
  }
  
  fullTrainFit = ctree(trainLabels~., data = trainData, control = ctrl)
  trainPred = predict(fullTrainFit, newdata = trainData)
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFit, newdata = testData)
  testMSE[i] = mse(testLabels, testPred)
  
}
## cvMSE =  9.752683 9.752683 9.752683 9.788350 9.893775
## trainMSE = 7.143705 7.143705 7.143705 7.143705 7.434476
## testMSE = 9.743890 9.743890 9.743890 9.743890 9.756172

## CV Folds on Max Depth (max depth = 30)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
maxDepth = c(3,7,15,24,30)

for (i in 1:5) {
  ctrl = ctree_control(maxdepth = maxDepth[i])
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = ctree(currLabels ~., data = currData, method = "anova", control = ctrl)
    
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFit, newdata = currTestData)
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
    
  }
  
  fullTrainFit = ctree(trainLabels~., data = trainData, control = ctrl)
  trainPred = predict(fullTrainFit, newdata = trainData)
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFit, newdata = testData)
  testMSE[i] = mse(testLabels, testPred)
  
}

## cvMSE =  15.703803 10.320838  9.752683  9.752683  9.752683
## trainMSE = 12.892299  7.407208  7.143705  7.143705  7.143705
## testMSE = 13.19903  9.46997  9.74389  9.74389  9.74389


##  CV Folds on Min # of observations per leaf node (Choose 1)
cvMSE = c(0,0,0,0,0)
trainMSE = c(0,0,0,0,0)
testMSE = c(0,0,0,0,0)
minBucket = c(1,3,7,15,30)

for (i in 1:5) {
  ctrl = ctree_control(minbucket = minBucket[i])
  for (j in 1:5) {
    currFold = cvTrainList[[j]]
    currLabels = currFold[,which(names(currFold)=="PCT_OBESE_ADULTS13")]
    currData = currFold[,-which(names(currFold)=="PCT_OBESE_ADULTS13")]
    tempFit = ctree(currLabels ~., data = currData, method = "anova", control = ctrl)
    
    currTestFold = cvTestList[[j]]
    currTestLabels = currTestFold[,which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    currTestData = currTestFold[,-which(names(currTestFold)=="PCT_OBESE_ADULTS13")]
    tempPred = predict(tempFit, newdata = currTestData)
    cvMSE[i] = cvMSE[i] + (mse(currTestLabels, tempPred)/5)
    
  }
  
  fullTrainFit = ctree(trainLabels~., data = trainData, control = ctrl)
  trainPred = predict(fullTrainFit, newdata = trainData)
  trainMSE[i] = mse(trainLabels, trainPred)
  testPred = predict(fullTrainFit, newdata = testData)
  testMSE[i] = mse(testLabels, testPred)
  
}


## cvMSE =   9.672926  9.685150  9.752683  9.701050 10.592628
## trainMSE = 7.100411 7.111610 7.143705 7.242246 8.261323
## testMSE = 9.612012  9.557788  9.743890  9.557493 10.117113



ctrl = ctree_control(minsplit =5,minbucket = 1, maxdepth = 30)
finalPartyFit = ctree(trainLabels ~ .,data = trainData, method = "anova", control = ctrl)

finalPartyTrainPred = predict(finalPartyFit,trainData)
trainMSE = mse(trainLabels, finalPartyTrainPred)

finalPartyTestPred = predict(finalPartyFit,testData)
testMSE = mse(testLabels, finalPartyTestPred)

plot(finalPartyFit, gp = gpar(fontsize = 6), inner_panel=node_inner,ip_args=list(abbreviate = TRUE, id = TRUE))
plot(finalPartyFit)
