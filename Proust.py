
import pandas as pd
import re
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en import English
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

tqdm.pandas()

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
    
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"addendum|consultant|GMC|Report date|\*{4}preliminary on call report\*{4}|reported by", "", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"consultant review", "", text)
            text = re.sub(r"provisional report", "", text)
            text = re.sub(r"agree with report", "", text)
            text = re.sub(r"i agree with", "", text)
            text = re.sub(r"report", "", text)
            text = re.sub(r"consultant radiologist", "", text)
            text = re.sub(r"consultant", "", text)
            text = re.sub(r"radiologist", "", text)
            text = re.sub(r"\n", "", text)
            text = re.sub(r"agreed", "", text)
            text = re.sub(r"entered", "", text)
            text = re.sub(r"by", "", text)
            text = re.sub(r"date", "", text)
            text = re.sub(r"please review", "", text)
            text = re.sub(r"review", "", text)
            text = re.sub(r"is made", "", text)
            text = re.sub(r"report", "", text)
            text = re.sub(r"ct Head", "", text)

            # Remove punctuation
            text = re.sub(r"[^\w\s]", "", text)
        
        else:
            text = str(text)

        return text
    
    df_cleaned['Clinical history'] = df_cleaned['Clinical history'].progress_apply(clean_text)
    df_cleaned['Report text'] = df_cleaned['Report text'].progress_apply(clean_text)

    def preprocess_text(text):
        # lowercase the text
        text = text.lower()
        # remove punctuation and whitespace
        text = re.sub(r'[^\w\s]', '', text)
        # remove stop words
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        # lemmatize the tokens
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        # join the lemmas into a string
        text = ' '.join(lemmas)
        return text

    df_cleaned['Clinical history'] = df_cleaned['Clinical history'].progress_apply(clean_text)
    df_cleaned['Report text'] = df_cleaned['Report text'].progress_apply(clean_text)

    def summarize_text(text):
        # Parse the text with spacy
        doc = nlp(text)

        # Get sentences with scores
        sentence_scores = {}
        for sent in doc.sents:
            # Score each sentence by summing the scores of its tokens
            score = sum([token.vector_norm for token in sent if not token.is_stop])
            sentence_scores[sent] = score

        # Get top 3 sentences by score
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

        # Combine the top sentences into a summary
        summary = ' '.join([sent.text for sent in top_sentences])

        return summary

    df_cleaned['Clinical history summary'] = df_cleaned['Clinical history'].progress_apply(summarize_text)
    df_cleaned['Report text summary'] = df_cleaned['Report text'].progress_apply(summarize_text)

    def extract_named_entities(text):
        # Parse the text with spaCy
        doc = nlp(text)

        # Extract named entities
        named_entities = [ent.text for ent in doc.ents]

        # Filter out duplicates and join the entities into a string
        unique_entities = ' '.join(list(set(named_entities)))

        return unique_entities

    # Apply the function to the 'Report text' column
    df_cleaned['Report named entities'] = df_cleaned['Report text'].progress_apply(extract_named_entities)

    def perform_lda_topic_modeling_and_assign(data, n_topics=5, n_top_words=10):
        # Create a CountVectorizer object for extracting features from the text
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        
        # Transform the text data into a document-term matrix
        dtm = vectorizer.fit_transform(data)
        
        # Create and fit the LDA model
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_model.fit(dtm)
        
        # Print the top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f"Topic #{topic_idx + 1}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

        # Assign topics to documents
        document_topics = lda_model.transform(dtm)
        assigned_topics = document_topics.argmax(axis=1)

        return assigned_topics

    # Perform LDA topic modeling on the 'Report text' column and assign topics to documents
    df_cleaned['Report topics'] = perform_lda_topic_modeling_and_assign(df_cleaned['Report text'], n_topics=5, n_top_words=10)

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
