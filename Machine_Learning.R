# Machine Learning
# Decision Tree
set.seed(42)
trainingDataDT <- read.csv("train_resume_data.csv")
testDataDT <-read.csv("test_resume_data.csv")
DataDF <-trainingDataDT[sample(nrow(trainingDataDT),replace = FALSE),]

DataDF$recruiter_decision <-factor(DataDF$recruiter_decision)
table(DataDF$recruiter_decision)

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
control <-trainControl(method = "repeatedcv", number = 10,savePredictions = TRUE)
model <-train(recruiter_decision~.,method="knn",data=trainingDataDT,trControl=control)
set.seed(123)
cv<-model$pred
cv

#Classification Evaluation
DataDF <- data[,c(2:32)]
set.seed(42)
partition <-createDataPartition(data$recuiter_decision,p=0.7, list=FALSE)
trainingData <-DataDF[partition]
testDataDT <-data[-partition,]
model <- train(recruiter_decision~., method="C5.0", data= trainingData)
model
customGrid <-expand.grid(model=c("rules", "tree"), trials = c(20, 25,30,35,40), winnow=c(FALSE))
model <-train(recruiter_decision~.,method="C5.0",DataDF=trainingDataDT,trainingGrid=customGrid)
model


