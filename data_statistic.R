install.packages("psych")
library(psych)
dataframe <- read.csv("resume_data_explored.csv")
set.seed(42)
cor(dataframe[c("experience_years", "recruiter_decision", "salary_expectation_.", "projects_count")], method = "spearman")
png(filename = "images/correlation_plot.png", width = 8, height = 6,units = "in", res = 300)
correlation_plot <- pairs(dataframe[c("experience_years", "recruiter_decision", "salary_expectation_.", "projects_count")])
dev.off()
png(filename = "images/splom_plot.png", width = 8, height = 6,units = "in", res = 300)
splom_plot <- pairs.panels(dataframe[c("experience_years", "recruiter_decision", "salary_expectation_.", "projects_count")], method="spearman")
dev.off()
non_encoded_features <- c("experience_years", "salary_expectation_.", "projects_count")
dataframe[non_encoded_features] <- scale(dataframe[non_encoded_features])
summary(dataframe[non_encoded_features])
install.packages("caTools")
library(caTools)
train_ratio <- 0.7
split_data <- sample.split(dataframe$recruiter_decision, SplitRatio = train_ratio)
train_data <- subset(dataframe, split_data == TRUE )
test_data <- subset(dataframe, split_data == FALSE )
table(test_data$recruiter_decision)
table(dataframe$recruiter_decision)

train_data_ai_researcher <- subset(train_data, job_role_ai_researcher == 1 )
train_data_cybersecurity_analyst <- subset(train_data, job_role_cybersecurity_analyst == 1 )
train_data_data_scientist <- subset(train_data, job_role_data_scientist == 1 )
train_data_software_engineer <- subset(train_data, job_role_software_engineer == 1 )
table(train_data_ai_researcher$recruiter_decision)
table(train_data_cybersecurity_analyst$recruiter_decision)
table(train_data_data_scientist$recruiter_decision)
table(train_data_software_engineer$recruiter_decision)

install.packages("smotefamily")
library(smotefamily)
smote_ai_researcher <- SMOTE(recruiter_decision ~ ., data=train_data_ai_researcher)
smote_cybersecurity_analyst <- SMOTE(recruiter_decision ~ ., data=train_data_cybersecurity_analyst)
smote_data_scientist <- SMOTE(recruiter_decision ~ ., data=train_data_data_scientist )
smote_software_engineer <- SMOTE(recruiter_decision ~ ., data=train_data_software_engineer)
table(smote_ai_researcher$recruiter_decision)
table(smote_cybersecurity_analyst$recruiter_decision)
table(smote_data_scientist$recruiter_decision)
table(smote_software_engineer$recruiter_decision)
















