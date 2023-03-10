
import pandas as pd
import re
import matplotlib.pyplot as plt
import scipy.stats as stats


def clean_data(file_path):
    
    # Read Excel file into a DataFrame
    df = pd.read_excel(file_path, header=2)

    # Define function to anonymize DataFrame
    def anonymize(df):
        # Select only the desired columns
        columns_to_keep = ['Sex', 'Ethnic origin', 'DOB', 'Age At Event', 'Vetted date', 'Vetted time', 'Event Date', 'Time', 'Examination', 'Exam Name', 'Clinical history', 'Report text']
        df = df[columns_to_keep]

        # Return the anonymized dataframe
        return df

    # Specify columns to include and drop rows with missing values in specified columns
    df_anonymized = anonymize(df)
    df_anonymized = df_anonymized.dropna(axis=0, how='any', subset=['Age At Event', 'Vetted date', 'Vetted time', 'Event Date', 'Time'])

    # Format and merge date and time columns for vetting and event time
    def merge_columns(df):
        df['Vetted date'] = pd.to_datetime(df['Vetted date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%d/%m/%Y')
        df['Vetted time'] = df['Vetted time'].fillna(0).astype(int).astype(str).str.zfill(4)
        df['Vetted time'] = df['Vetted time'].str[:-2] + ':' + df['Vetted time'].str[-2:]
        df['Vetted datetime'] = pd.to_datetime(df['Vetted date'] + ' ' + df['Vetted time'], format='%d/%m/%Y %H:%M', errors='coerce')

        df['Event Date'] = pd.to_datetime(df['Event Date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%d/%m/%Y')
        df['Time'] = df['Time'].fillna(0).astype(int).astype(str).str.zfill(4)
        df['Time'] = df['Time'].str[:-2] + ':' + df['Time'].str[-2:]
        df['Event datetime'] = pd.to_datetime(df['Event Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
        df.drop(['Vetted date', 'Vetted time', 'Event Date', 'Time'], axis=1, inplace=True)

        return df

    df_cleaned = merge_columns(df_anonymized)

    def test_sex_significance(df):
        ct_by_sex = pd.crosstab(df['Sex'], columns='count')
        ct_by_sex.index = ['Female', 'Male']
        chi2, p, dof, expected = stats.chi2_contingency(ct_by_sex)

        if p < 0.05:
            print(f"There is a significant association between sex and the number of CT scans (p-value = {p}).")
        else:
            print(f"There is no significant association between sex and the number of CT scans (p-value = {p}).")

    def test_age_significance(df):
        age_groups = pd.cut(df['Age At Event'], bins=[0, 18, 30, 40, 50, 60, 70, 120])
        ct_by_age = pd.crosstab(age_groups, columns='count')
        f_statistic, p_value = stats.f_oneway(*[df.loc[age_groups == g, 'Age At Event'] for g in ct_by_age.index])

        if p_value < 0.05:
            print(f"There is a significant difference in the mean number of CT scans across age groups (p-value = {p_value}).")
        else:
            print(f"There is no significant difference in the mean number of CT scans across age groups (p-value = {p_value}).")
       

    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r"ADDENDUM|consultant|GMC|Report date|\*{4}Preliminary On Call Report\*{4}|Reported By", "", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"Consultant Review", "", text)
            text = re.sub(r"provisional report", "", text)
            text = re.sub(r"Agree with report", "", text)
            text = re.sub(r"I agree with", "", text)
            text = re.sub(r"report", "", text)
            text = re.sub(r"Consultant radiologist", "", text)
            text = re.sub(r"Consultant", "", text)
            text = re.sub(r"radiologist", "", text)
            text = re.sub(r"\n", "", text)
            text = re.sub(r"Agreed", "", text)
            text = re.sub(r"Entered by", "", text)
            text = re.sub(r"by", "", text)
            text = re.sub(r"Date", "", text)
            text = re.sub(r"Please review", "", text)
            text = re.sub(r"review", "", text)
            text = re.sub(r"is made", "", text)
            text = re.sub(r"Report", "", text)
            text = re.sub(r"CT Head", "", text)

            # Remove punctuation
            text = re.sub(r"[^\w\s]", "", text)

        return text

    df_cleaned['Clinical history'] = df_cleaned['Clinical history'].apply(clean_text)
    df_cleaned['Report text'] = df_cleaned['Report text'].apply(clean_text)

    # Print total number of scans
    total_scans = len(df_cleaned)
    print(f"Total number of scans: {total_scans}")


    def plot_scans_by_sex(df):
        df['Sex'].value_counts().plot(kind='bar')
        plt.title('Number of scans by Sex')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.show()

        test_sex_significance(df)

    def plot_scans_by_age(df):
        df['Age At Event'] = df['Age At Event'].astype(int)
        df['Age At Event'].plot(kind='hist', bins=20)
        plt.title('Number of scans by Age')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.show()

        test_age_significance(df)
        
    def test_sex_significance(df):
        ct_by_sex = pd.crosstab(df['Sex'], columns='count')
        ct_by_sex.index = ['Female', 'Male']
        chi2, p, dof, expected = stats.chi2_contingency(ct_by_sex)

        if p < 0.05:
            print(f"There is a significant association between sex and the number of CT scans (p-value = {p}).")
        else:
            print(f"There is no significant association between sex and the number of CT scans (p-value = {p}).")

    def test_age_significance(df):
        age_groups = pd.cut(df['Age At Event'], bins=[0, 18, 30, 40, 50, 60, 70, 120])
        ct_by_age = pd.crosstab(age_groups, columns='count')
        f_statistic, p_value = stats.f_oneway(*[df.loc[age_groups == g, 'Age At Event'] for g in ct_by_age.index])

        if p_value < 0.05:
            print(f"There is a significant difference in the mean number of CT scans across age groups (p-value = {p_value}).")
        else:
            print(f"There is no significant difference in the mean number of CT scans across age groups (p-value = {p_value}).")

    # Plot number of scans by day, month, and year of vetting datetime
    df_cleaned['Vetted datetime'].dt.day.value_counts().sort_index().plot(kind='bar')
    plt.title('Number of scans by Day of Vetting Datetime')
    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.show()

    df_cleaned['Vetted datetime'].dt.month.value_counts().sort_index().plot(kind='bar')
    plt.title('Number of scans by Month of Vetting Datetime')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.show()

    df_cleaned['Vetted datetime'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Number of scans by Year of Vetting Datetime')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.show()

    # Return the cleaned DataFrame
    return df_cleaned

    plot_scans_by_sex(df_cleaned)
    plot_scans_by_age(df_cleaned)




