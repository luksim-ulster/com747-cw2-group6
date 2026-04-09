import pandas as pd

# import dataset
dataframe = pd.read_csv("AI_Resume_Screening.csv")

# initial look
print(f"{dataframe.head(5).to_string()}\n")
dataframe.info()
print(f"\nluksim note - observed that Name is personally identifiable")
print(f"luksim note - observed that Certifications heading has missing values\n")

# drop personal information
dataframe = dataframe.drop(columns=['Name'])

# find missing Certifications rows
rows_with_nulls = dataframe['Certifications'].isna()
# current state
print(dataframe[rows_with_nulls].head(3).to_string())
# replace
dataframe['Certifications'] = dataframe['Certifications'].fillna('None')
# check update
print(dataframe[rows_with_nulls].head(3).to_string())
dataframe.info()
print(f"\nluksim note - observed that column headings aren't standardised\n")

# tidy headings names
dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(r'[\(\)]', '', regex=True).str.replace('-', '_').str.replace(' ', '_')
dataframe.info()
print(f"\nluksim note - observed that recruiter_decision str needs to be mapped to int64")
print(f"luksim note - observed that skills, education, certifications, and job_role are str and need to be mapped to int64\n")

# binary encoding
dataframe['recruiter_decision'] = dataframe['recruiter_decision'].map({'Hire': 1, 'Reject': 0})

# one hot encoding
one_hot_skills = dataframe['skills'].str.strip().str.lower().str.get_dummies(sep=', ').add_prefix('skill_')
one_hot_skills.columns = one_hot_skills.columns.str.replace(' ', '_')
one_hot_education = dataframe['education'].str.strip().str.lower().str.get_dummies(sep=', ').add_prefix('education_')
one_hot_education.columns = one_hot_education.columns.str.replace('.', '')
one_hot_role = dataframe['job_role'].str.strip().str.lower().str.get_dummies(sep=', ').add_prefix('job_role_')
one_hot_role.columns = one_hot_role.columns.str.replace(' ', '_')
one_hot_certification = dataframe['certifications'].str.strip().str.lower().str.get_dummies(sep=', ').add_prefix('certification_')
one_hot_certification.columns = one_hot_certification.columns.str.replace(' ', '_')
# remove clutter columns
one_hot_certification = one_hot_certification.drop(columns=['certification_none'])
# remove encoded headings from dataframe
dataframe = dataframe.drop(columns=['skills', 'education', 'job_role', 'certifications'])

# check processing
print(f"{dataframe.head(3).to_string()}\n")
print(f"{one_hot_skills.head(3).to_string()}\n")
print(f"{one_hot_education.head(3).to_string()}\n")
print(f"{one_hot_role.head(3).to_string()}\n")
print(f"{one_hot_certification.head(3).to_string()}\n")

# combine
dataframe = pd.concat([dataframe, one_hot_skills, one_hot_education, one_hot_role, one_hot_certification], axis=1)
# save to csv
dataframe.to_csv('ai_resume_data_processed.csv', index=False)

print(f"luksim note - saved to ai_resume_data_processed.csv and ready for analysis\n")
