# load and install packages
install.packages("moments")
library(moments)
install.packages("ggplot2")
library(ggplot2)
install.packages("effectsize")
library(effectsize)

# load processed dataframe
dataframe <- read.csv("resume_data_processed.csv")

# ensure reproducibility
set.seed(42)

# check for missing values
sum(is.na(dataframe))
print("luksim note - no missing values")

# check datatypes
str(dataframe)
print("luksim note - all int types")

# summary statistics
summary(dataframe$experience_years)
summary(dataframe$salary_expectation_.)
summary(dataframe$projects_count)
print("luksim note - data is symetrically distributed")
print("luksim note - data make sense and contains no extreme values")
n <- length(dataframe$salary_expectation_.)
print(paste("salary_expectation_. SD:", sd(dataframe$salary_expectation_.)))
print(paste("salary_expectation_. SE:", sd(dataframe$salary_expectation_.)/sqrt(n)))
n <- length(dataframe$projects_count)
print(paste("projects_count SD:", sd(dataframe$projects_count)))
print(paste("projects_count SE:", sd(dataframe$projects_count)/sqrt(n)))
n <- length(dataframe$experience_years)
print(paste("experience_years SD:", sd(dataframe$experience_years)))
print(paste("experience_years SE:", sd(dataframe$experience_years)/sqrt(n)))
print("luksim note - data is spread out")
print("luksim note - data is accurate")

# categorical imbalance
# by recruiter decision (target variable)
table(dataframe$recruiter_decision)
print("luksim note - large imbalance in recruiter decision, SMOTE is needed - action needed")
# by job role
colSums(dataframe[, c("job_role_ai_researcher", "job_role_cybersecurity_analyst", "job_role_data_scientist", "job_role_software_engineer")])
print("luksim note - job roles are balanced, SMOTE should be applied per job role - action needed")

# frequency of skill
colSums(dataframe[, 5:18])
# frequency of education
colSums(dataframe[, 19:23])
# frequency of certification
colSums(dataframe[, 28:30])
print("luksim note - no zero variance predictors in skill, education, or certification features")

# normality test
shapiro.test(dataframe$salary_expectation_.)
shapiro.test(dataframe$projects_count)
shapiro.test(dataframe$experience_years)
print("luksim note - salary, project, and experience data are not normally distributed")

# skewness and kurtosis
print(paste("salary_expectation_. skewness:", skewness(dataframe$salary_expectation_.)))
print(paste("salary_expectation_. kurtosis:", kurtosis(dataframe$salary_expectation_.)))
print(paste("projects_count skewness:", skewness(dataframe$projects_count)))
print(paste("projects_count kurtosis:", kurtosis(dataframe$projects_count)))
print(paste("experience_years skewness:", skewness(dataframe$experience_years)))
print(paste("experience_years kurtosis:", kurtosis(dataframe$experience_years)))
print("luksim note - skewness suggests data is symmetrical and evenly distributed around the mean")
print("luksim note - kurtosis suggests data is platykurtic with flat tops and thin tails/no extreme outliers")

# wilcoxon rank sum test since not normally distributed
print(wilcox.test(salary_expectation_. ~ recruiter_decision, data = dataframe))
print("luksim note - no statistical difference in salary between hired and rejected candidates")
print(rank_biserial(salary_expectation_. ~ recruiter_decision, data = dataframe))
print("luksim note - tiny effect size")

print(wilcox.test(projects_count ~ recruiter_decision, data = dataframe))
print("luksim note - statistical difference in projects between hired and rejected candidates")
print(rank_biserial(projects_count ~ recruiter_decision, data = dataframe))
print("luksim note - medium/large effect size")

print(wilcox.test(experience_years ~ recruiter_decision, data = dataframe))
print("luksim note - statistical difference in experience between hired and rejected candidates")
print(rank_biserial(experience_years ~ recruiter_decision, data = dataframe))
print("luksim note - large effect size")

# density and histogram plots
# years of experience
experience_distribution <- ggplot(dataframe, aes(x = experience_years)) +
  geom_histogram(aes(y = after_stat(density)), bins = 11, color = "black", alpha = 0.1) +
  geom_density(color = "black", size = 1) +
  labs(title = "Experience Distribution", x = "Experience (Years)", y = "Density") +
  theme_minimal()
ggsave(filename = "images/experience_distribution.png", plot = experience_distribution, width = 8, height = 6, dpi = 300, bg = "white")

# salary expectation
salary_expectation_distribution <- ggplot(dataframe, aes(x = salary_expectation_.)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, color = "black", alpha = 0.1) +
  geom_density(color = "black", size = 1) +
  labs(title = "Salary Expectation Distribution", x = "Salary Expectation ($)", y = "Density") +
  theme_minimal()
ggsave(filename = "images/salary_expectation_distribution.png", plot = salary_expectation_distribution, width = 8, height = 6, dpi = 300, bg = "white")

# projects count
projects_distribution <- ggplot(dataframe, aes(x = projects_count)) +
  geom_histogram(aes(y = after_stat(density)), bins = 11, color = "black", alpha = 0.1) +
  geom_density(color = "black", size = 1) +
  labs(title = "Projects Distribution", x = "Projects (Count)", y = "Density") +
  theme_minimal()
ggsave(filename = "images/projects_distribution.png", plot = projects_distribution, width = 8, height = 6, dpi = 300, bg = "white")

print("luksim note - density and histogram plots shows data is fairly evenly distributed")

# add job role category
dataframe$role_category <- "Unknown"
dataframe$role_category[dataframe$job_role_ai_researcher == 1] <- "AI Researcher"
dataframe$role_category[dataframe$job_role_cybersecurity_analyst == 1] <- "Cybersecurity Analyst"
dataframe$role_category[dataframe$job_role_data_scientist == 1] <- "Data Scientist"
dataframe$role_category[dataframe$job_role_software_engineer == 1] <- "Software Engineer"
dataframe$role_category <- as.factor(dataframe$role_category)

# boxplots
# years of experience
experience_boxplot <- ggplot(dataframe, aes(x = role_category, y = experience_years)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.25, height = 0, size = 0.8) +
  theme_minimal() +
  labs(title = "Experience Boxplot by Job Role", x = "Job Role", y = "Experience (Years)") +
  theme(legend.position = "none")
ggsave(filename = "images/experience_boxplot.png", plot = experience_boxplot, width = 8, height = 6, dpi = 300, bg = "white")

outliers_experience <- boxplot.stats(dataframe$experience_years)$out
print(outliers_experience)
print(paste("experience_years outliers:", length(outliers_experience)))

# salary expectation
salary_expectation_boxplot <- ggplot(dataframe, aes(x = role_category, y = salary_expectation_.)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.25, height = 0, size = 0.8) +
  theme_minimal() +
  labs(title = "Salary Expectation Boxplot by Job Role", x = "Job Role", y = "Salary Expectation ($)") +
  theme(legend.position = "none")
ggsave(filename = "images/salary_expectation_boxplot.png", plot = salary_expectation_boxplot, width = 8, height = 6, dpi = 300, bg = "white")

outliers_salary <- boxplot.stats(dataframe$salary_expectation_.)$out
print(outliers_salary)
print(paste("salary_expectation_. outliers:", length(outliers_salary)))

# projects count
projects_boxplot <- ggplot(dataframe, aes(x = role_category, y = projects_count)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.25, height = 0, size = 0.8) +
  theme_minimal() +
  labs(title = "Projects Boxplot by Job Role", x = "Job Role", y = "Projects (Count)") +
  theme(legend.position = "none")
ggsave(filename = "images/projects_boxplot.png", plot = projects_boxplot, width = 8, height = 6, dpi = 300, bg = "white")

outliers_projects <- boxplot.stats(dataframe$projects_count)$out
print(outliers_projects)
print(paste("projects_count outliers:", length(outliers_projects)))

print("luksim note - box plots shows no outliers so no further data pruning needed")
print("luksim note - experience, salary, and projects are similar across roles")

# drop job role category (can be readded for SMOTE)
dataframe$role_category <- NULL


