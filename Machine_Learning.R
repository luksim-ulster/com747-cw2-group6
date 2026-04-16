# Machine Learning
# Decision Tree
set.seed(42)
# luksim - need to rerun with latest main update as currently index included in train dataset
trainingDataDT <- read.csv("train_resume_data.csv")
testDataDT <-read.csv("test_resume_data.csv")
DataDF <-trainingDataDT[sample(nrow(trainingDataDT),replace = FALSE),] # luksim - done already, can be removed

DataDF$recruiter_decision <-factor(DataDF$recruiter_decision) # luksim - do for both training and testing datasets
table(DataDF$recruiter_decision)

# luksim - why not using the testDataDT? data is already split, can remove two lines below
trainingDataDT <- DataDF[c(1:500),]
testDataDT <- DataDF[c(501:nrow(DataDF)),]

install.packages("C50")
library(C50)
decisionTree <-C5.0(recruiter_decision~.,data=trainingDataDT)
summary(decisionTree)
plot(decisionTree)
# K-Fold cross - validation
install.packages("caret")
library(caret)
control <-trainControl(method = "repeatedcv", number = 10,savePredictions = TRUE) # luksim - can also add repeats = x to improve
model <-train(recruiter_decision~.,method="knn",data=trainingDataDT,trControl=control) # luksim - method should be C5.0 not knn for decision tree
set.seed(123) # luksim - done on line 3 already, can be removed
cv<-model$pred
cv

#Classification Evaluation
# luksim - I don't think any of the below is needed for classification evaluation, line 40 can be moved up and referenced in model train()
# I think we just need to run the model with the test data to get predictions and compare (?predict.C5.0 and ?confusionMatrix)
DataDF <- data[,c(2:32)]
set.seed(42)
partition <-createDataPartition(data$recuiter_decision,p=0.7, list=FALSE)
trainingData <-DataDF[partition]
testDataDT <-data[-partition,]
model <- train(recruiter_decision~., method="C5.0", data= trainingData)
model
customGrid <-expand.grid(model=c("rules", "tree"), trials = c(20, 25,30,35,40), winnow=c(FALSE)) # luksim - I think this can be added to line 25, tuneGrid = customGrid
model <-train(recruiter_decision~.,method="C5.0",DataDF=trainingDataDT,trainingGrid=customGrid)
model


