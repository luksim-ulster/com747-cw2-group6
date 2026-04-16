# Machine Learning
# Decision Tree
set.seed(42)

trainingDataDT <- read.csv("train_resume_data.csv")
testDataDT <-read.csv("test_resume_data.csv")


trainingDataDT$recruiter_decision <-factor(trainingDataDT$recruiter_decision)
testDataDT$recruiter_decision <-factor(testDataDT$recruiter_decision)
table(testDataDT$recruiter_decision)




install.packages("C50")
library(C50)
decisionTree <-C5.0(recruiter_decision~.,data=trainingDataDT)
summary(decisionTree)
plot(decisionTree)
# K-Fold cross - validation
install.packages("caret")
library(caret)
control <-trainControl(method = "repeatedcv", number = 10,savePredictions = TRUE,repeats = 3) 
customGrid <-expand.grid(model=c("rules", "tree"), trials = c(20, 25,30,35,40), winnow=c(FALSE))
model <-train(recruiter_decision~.,method="C5.0",data=trainingDataDT,trControl=control,tuneGrid= customGrid)

predictions <-predict(model, newdata=testDataDT)
confusionMatrix(predictions, testDataDT$recruiter_decision)

install.packages("MLmetrics")
library(MLmetrics)
F1_Score(predictions,testDataDT$recruiter_decision)