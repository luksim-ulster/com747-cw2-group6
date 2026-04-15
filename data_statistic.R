install.packages("psych")
library(psych)
dataframe <- read.csv("resume_data_explored.csv")
set.seed(42)
non_encoded_features <- c("experience_years", "salary_expectation_.", "projects_count")

cor(dataframe[c(non_encoded_features, "recruiter_decision")], method = "spearman")
png(filename = "images/correlation_plot.png", width = 8, height = 6,units = "in", res = 300)
correlation_plot <- pairs(dataframe[c(non_encoded_features, "recruiter_decision")])
dev.off()
png(filename = "images/splom_plot.png", width = 8, height = 6,units = "in", res = 300)
splom_plot <- pairs.panels(dataframe[c(non_encoded_features, "recruiter_decision")], method="spearman")
dev.off()
# dataframe[non_encoded_features] <- scale(dataframe[non_encoded_features])
summary(dataframe[non_encoded_features])
install.packages("caTools")
library(caTools)
train_ratio <- 0.7
split_data <- sample.split(dataframe$recruiter_decision, SplitRatio = train_ratio)
train_data <- subset(dataframe, split_data == TRUE )
test_data <- subset(dataframe, split_data == FALSE )

# scale data
install.packages("caret")
library(caret)
scaling <- preProcess(train_data[non_encoded_features], method = c("center", "scale"))
train_data[non_encoded_features] <- predict(scaling, train_data[non_encoded_features])
test_data[non_encoded_features]  <- predict(scaling, test_data[non_encoded_features])
print("luksim note - split done first to avoid data leakage from test into train set during standardisation")

# function to print recruiter decision balance table and ratio
get_decision_table_ratio <- function(name, data){
  decision_table <- table(data$recruiter_decision)
  print(decision_table)
  print(paste(name, "- reject:hire ratio:", decision_table[1] / decision_table[2]))
}

get_decision_table_ratio("test_data", test_data)
get_decision_table_ratio("dataframe", dataframe)

# categorise target feature
train_data$recruiter_decision <- as.factor(train_data$recruiter_decision)

# split train set by role for smote
train_data_ai_researcher <- subset(train_data, job_role_ai_researcher == 1 )
train_data_cybersecurity_analyst <- subset(train_data, job_role_cybersecurity_analyst == 1 )
train_data_data_scientist <- subset(train_data, job_role_data_scientist == 1 )
train_data_software_engineer <- subset(train_data, job_role_software_engineer == 1 )

# imbalance by role
get_decision_table_ratio("train_data_ai_researcher", train_data_ai_researcher)
get_decision_table_ratio("train_data_cybersecurity_analyst", train_data_cybersecurity_analyst)
get_decision_table_ratio("train_data_data_scientist", train_data_data_scientist)
get_decision_table_ratio("train_data_software_engineer", train_data_software_engineer)
print("luksim note - recruiter decision needs to be balanced by role to avoid feature data contamination across roles")

# smote by role
install.packages("smotefamily")
library(smotefamily)

features_ai_researcher <- subset(train_data_ai_researcher, select = -recruiter_decision)
target_ai_researcher <- train_data_ai_researcher$recruiter_decision
smote_ai_researcher <- SMOTE(features_ai_researcher, target_ai_researcher)$data
names(smote_ai_researcher)[names(smote_ai_researcher) == "class"] <- "recruiter_decision"

features_cybersecurity_analyst <- subset(train_data_cybersecurity_analyst, select = -recruiter_decision)
target_cybersecurity_analyst <- train_data_cybersecurity_analyst$recruiter_decision
smote_cybersecurity_analyst <- SMOTE(features_cybersecurity_analyst, target_cybersecurity_analyst)$data
names(smote_cybersecurity_analyst)[names(smote_cybersecurity_analyst) == "class"] <- "recruiter_decision"

features_data_scientist <- subset(train_data_data_scientist, select = -recruiter_decision)
target_data_scientist <- train_data_data_scientist$recruiter_decision
smote_data_scientist <- SMOTE(features_data_scientist, target_data_scientist)$data
names(smote_data_scientist)[names(smote_data_scientist) == "class"] <- "recruiter_decision"

features_software_engineer <- subset(train_data_software_engineer, select = -recruiter_decision)
target_software_engineer <- train_data_software_engineer$recruiter_decision
smote_software_engineer <- SMOTE(features_software_engineer, target_software_engineer)$data
names(smote_software_engineer)[names(smote_software_engineer) == "class"] <- "recruiter_decision"

# balanced by role
get_decision_table_ratio("smote_ai_researcher", smote_ai_researcher)
get_decision_table_ratio("smote_cybersecurity_analyst", smote_cybersecurity_analyst)
get_decision_table_ratio("smote_data_scientist", smote_data_scientist)
get_decision_table_ratio("smote_software_engineer", smote_software_engineer)
print("luksim note - recruiter decision for each role is now balanced")

# merge and shuffle
smote_train_data <- rbind(smote_ai_researcher, smote_cybersecurity_analyst, smote_data_scientist, smote_software_engineer)
shuffled_indices <- sample(nrow(smote_train_data))
smote_train_data <- smote_train_data[shuffled_indices, ]
get_decision_table_ratio("smote_train_data", smote_train_data)
print("luksim note - recruiter decision now balanced across dataset")
print("luksim note - smote has introduced fractional values for expected binary features")

# round fractional values of binary features
encoded_features <- setdiff(names(smote_train_data), c(non_encoded_features, "recruiter_decision"))
smote_train_data[encoded_features] <- round(smote_train_data[encoded_features])

# boruta to find importance of features
install.packages("Boruta")
library(Boruta)
smote_train_data$recruiter_decision <- as.factor(smote_train_data$recruiter_decision)
boruta <- Boruta(recruiter_decision~., data=smote_train_data)
boruta_fixed <- TentativeRoughFix(boruta)
print(boruta_fixed)
png(filename = "images/boruta_plot.png", width = 8, height = 10, units = "in", res = 300)
par(mar = c(14, 4, 1, 1))
plot(boruta_fixed, las = 2, cex.axis = 0.8, xlab = "")
dev.off()
print(getSelectedAttributes(boruta_fixed, withTentative = FALSE))
print("luksim note - experience and projects have the most influence on the model")
print("luksim note - all features to be used in the model based on boruta results")

# summary statistics of finalised train data
summary(smote_train_data)

# save train and test datasets for modelling
write.csv(smote_train_data, file = "train_resume_data.csv", row.names = FALSE)
write.csv(test_data, file = "test_resume_data.csv", row.names = FALSE)
