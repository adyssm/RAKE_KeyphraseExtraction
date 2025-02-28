import pandas as pd
import itertools
from collections import defaultdict

class Model:
    def build_cooccurrence_matrix_by_words(self, df, column_name='stopword_removed'):
        cooccurrence_dict = defaultdict(int)
        for token_list in df[column_name]:
            for word1, word2 in itertools.combinations(token_list, 2):
                sorted_pair = tuple(sorted([word1, word2]))
                cooccurrence_dict[sorted_pair] += 1 
        return pd.DataFrame(cooccurrence_dict.items(), columns=['Word Pair', 'Co-occurrence'])

    def calculate_word_scores(self, df, cooccurrence_column='Co-occurrence', word_pair_column='Word Pair'):
        word_frequency = defaultdict(int)
        word_degree = defaultdict(int)
        for idx, row in df.iterrows():
            word_pair = row[word_pair_column]
            cooccur_count = row[cooccurrence_column]
            word1, word2 = word_pair
            word_frequency[word1] += cooccur_count
            word_frequency[word2] += cooccur_count
            word_degree[word1] += cooccur_count + 1
            word_degree[word2] += cooccur_count + 1
        word_scores = {word: word_degree[word] / word_frequency[word] for word in word_frequency}
        return pd.DataFrame({'Word': list(word_frequency.keys()), 'Frequency': list(word_frequency.values()), 'Degree': list(word_degree.values()), 'Ratio': [word_scores[word] for word in word_scores]})
    
    def calculate_keyphrase_scores(self, row, word_scores_df, ratio_column='Ratio'):
        keyphrase_tokens = row['noun_phrase_chunks']
        keyphrase_scores = []
        for keyphrase in keyphrase_tokens:
            total_score = 0
            words = keyphrase.split()
            for word in words:
                ratio = word_scores_df[word_scores_df['Word'] == word][ratio_column].values
                if len(ratio) > 0:
                    total_score += ratio[0]
            keyphrase_scores.append((keyphrase, total_score))
        return keyphrase_scores

    def apply_keyphrase_scoring(self, df, word_scores_df):
        df['Keyphrase_Scores'] = df.apply(self.calculate_keyphrase_scores, axis=1, word_scores_df=word_scores_df)
        return df
