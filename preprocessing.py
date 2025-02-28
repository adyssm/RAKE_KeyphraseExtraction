import pandas as pd
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class Preprocessing:
    def load_datasets_as_table(self, input_abstrak):
        return pd.DataFrame({'file_name': ['input_abstrak'], 'content': [input_abstrak], 'combined': [input_abstrak]})
    
    def case_folding(self, df):
        df['case_folding'] = df['combined'].str.lower()
        return df
    
    def cleaning_text(self, df):
        def clean_text(text):
            text = re.sub(r'[^a-z\s]', '', text)
            return re.sub(r'\s+', ' ', text).strip()
        df['cleaned'] = df['case_folding'].apply(clean_text)
        return df
    
    def tokenisasi_text(self, df):
        df['tokenized'] = df['cleaned'].apply(word_tokenize)
        return df
    
    def stopword_removal(self, df):
        factory = StopWordRemoverFactory()
        stopwords = set(factory.get_stop_words())
        df['stopword_removed'] = df['tokenized'].apply(lambda tokens: [word for word in tokens if word not in stopwords])
        return df
