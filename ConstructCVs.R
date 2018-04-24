setwd("/Users/joshuaogunleye/Desktop/CIS 520/Project/Obesity CV, Train, Test Data")
install.packages("caret")
library("caret")
data <- read.csv("CartImp.csv")
set.seed(1234)

sample = sample.int(n = nrow(data), size = floor(.80*nrow(data)), replace = F)

#80% train
trainData = data[sample,]
write.csv(trainData, file = "ObesityFullTrainSet.csv")

cvFolds = createFolds(trainData$PCT_OBESE_ADULTS13, k = 5, list = TRUE, returnTrain = FALSE)

for (i in 1:5) {
  write.csv(trainData[cvFolds[[i]],], file = sprintf("ObesityCVFold_%d.csv",i))
}


#20% test
testData = data[-sample,]
write.csv(testData, file = "ObesityTestSet.csv")

