
import pandas as pd
import re

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

    # Clean free text columns for text analysis
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
            text = re.sub(r"radiologist", "",
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

# Clean the 'Clinical history' and 'Report text' columns
df_cleaned = merge_columns(df_anonymized)
df_cleaned['Clinical history'] = df_cleaned['Clinical history'].apply(clean_text)
df_cleaned['Report text'] = df_cleaned['Report text'].apply(clean_text)

# Return the cleaned dataframe
return df_cleaned
