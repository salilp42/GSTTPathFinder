
import pandas as pd
import re
import matplotlib.pyplot as plt

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

    # Clean the 'Clinical history' and 'Report text' columns
    df_cleaned = merge_columns(df_anonymized)
    df_cleaned['Clinical history'] = df_cleaned['Clinical history'].apply(lambda x: re.sub(r'[^\w\s]|\d', '', x.lower()) if isinstance(x, str) else '')
    df_cleaned['Report text'] = df_cleaned['Report text'].apply(lambda x: re.sub(r'[^\w\s]|\d', '', x.lower()) if isinstance(x, str) else '')


    # Print total number of scans
    total_scans = len(df_cleaned)
    print(f"Total number of scans: {total_scans}")

    # Plot number of scans by Sex
    df_cleaned['Sex'].value_counts().plot(kind='bar')
    plt.title('Number of scans by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.show()

    # Plot number of scans by Age
    df_cleaned['Age At Event'] = df_cleaned['Age At Event'].astype(int)
    df_cleaned['Age At Event'].plot(kind='hist', bins=20)
    plt.title('Number of scans by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

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

