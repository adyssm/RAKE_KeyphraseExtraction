import os
from nltk.tag import CRFTagger

# Inisialisasi model Flair untuk POS tagging (Bahasa Indonesia)
class POSTagging:
    def __init__(self):

# Inisialisasi CRFTagger
        self.ct = CRFTagger()
# Path ke model yang sudah diunduh secara lokal
        self.ct.set_model_file('C:\SKRIPSI - Copy\post_tagger.model')  # Ganti dengan path model CRF yang sesuai


        # Fungsi untuk tagging POS menggunakan CRFTagger
    def tag_pos(self,tokens):
        pos_tags = self.ct.tag(tokens)  # Menggunakan CRFTagger untuk POS tagging
        return pos_tags


    # Fungsi untuk menerapkan POS tagging ke DataFrame
    def apply_pos_tagging(self, df):
        df['combined_pos_tagged'] = df['stopword_removed'].apply(self.tag_pos)
        return df


