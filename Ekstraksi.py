import streamlit as st
from preprocessing import Preprocessing
from pos_tagging import POSTagging
from np_chunking import POSTaggingChunking
from model import Model

class Main:
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.pos_tagging = POSTagging()
        self.np_chunking = POSTaggingChunking()
        self.model = Model()

    # Fungsi untuk meranking keyphrases dengan struktur tertentu
    def rank_keyphrases(self, keyphrase_scores):
        # Pisahkan frasa berdasarkan jumlah kata
        three_word_phrases = [kp for kp in keyphrase_scores if len(kp[0].split()) == 3]
        two_word_phrases = [kp for kp in keyphrase_scores if len(kp[0].split()) == 2]
        other_phrases = [kp for kp in keyphrase_scores if len(kp[0].split()) != 2 and len(kp[0].split()) != 3]
        
        # Urutkan setiap daftar berdasarkan skor (dari yang tertinggi)
        three_word_phrases = sorted(three_word_phrases, key=lambda x: x[1], reverse=True)
        two_word_phrases = sorted(two_word_phrases, key=lambda x: x[1], reverse=True)
        other_phrases = sorted(other_phrases, key=lambda x: x[1], reverse=True)

        # Ambil frasa berdasarkan struktur yang diinginkan
        top_keyphrases = three_word_phrases[:3] + two_word_phrases[:3]
        
        # Isi sisa slot dengan frasa lainnya hingga mencapai 15
        remaining_phrases = three_word_phrases[3:] + two_word_phrases[3:] + other_phrases
        top_keyphrases += remaining_phrases[:15 - len(top_keyphrases)]
        
        return top_keyphrases[:15]

    def run(self):
        st.set_page_config(page_title="Keyphrase Extraction System", layout="centered")
        
        # Layout Navbar menggunakan CSS dan HTML
        st.markdown("""
         <style>
              .main-container {
            background-color: #E6F7FF; /* Latar belakang biru pastel */
            padding: 20px;
            border-radius: 10px;
        }
        .main-header {
            padding: 20px;
            background-color: #F9C9A1; /* Warna latar opsional untuk header */
        }
        .header-title {
            font-size: 26px; /* Ukuran font */
            font-weight: bold; /* Teks tebal */
            color: #333; /* Warna teks */
            text-align: center; /* Rata tengah */
        }
    </style>
    <div class="main-header">
        <div class="header-title">Ekstraksi Kata Kunci pada Artikel Berbahasa Indonesia Menggunakan Metode <i>Rapid Automatic Keyphrase Extraction</i> (RAKE)</div>
    </div>
            <style>
                .navbar {
                    background-color: #F9C9A1;
                    padding: 10px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 2px solid #ccc;
                }
                .navbar-title {
                    font-size: 22px;
                    font-weight: bold;
                    color: #333;
                }
                .navbar-buttons {
                    display: flex;
                    gap: 15px;
                }
                .navbar-buttons a {
                    text-decoration: none;
                    font-weight: bold;
                    color: black;
                    background-color: #fff;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-size: 16px;
                    border: none;
                    outline: none;
                }
                .navbar-buttons a:hover {
                    background-color: #00008B;
                    color: white;
                }
            </style>
            </div>
        """, unsafe_allow_html=True)

        # Form Input
        with st.form(key='input_form'):
            judul = st.text_input("Input Judul", placeholder="Masukkan judul di sini...")
            abstrak = st.text_area("Input Abstrak", placeholder="Masukkan abstrak di sini...")
            submit_button = st.form_submit_button(label="SUBMIT")
        
        if submit_button:
            # Preprocessing
            combined_df = self.preprocessing.load_datasets_as_table(abstrak)
            processed_df = self.preprocessing.case_folding(combined_df)
            processed_df = self.preprocessing.cleaning_text(processed_df)
            processed_df = self.preprocessing.tokenisasi_text(processed_df)
            processed_df = self.preprocessing.stopword_removal(processed_df)
            
            # POS Tagging
            processed_df = self.pos_tagging.apply_pos_tagging(processed_df)
            
            # NP Chunking
            final_df = self.np_chunking.apply_np_chunking(processed_df)
            
            # Scoring Model
            cooccurrence_matrix_words = self.model.build_cooccurrence_matrix_by_words(final_df)
            word_scores_df = self.model.calculate_word_scores(cooccurrence_matrix_words)
            final_scored_df = self.model.apply_keyphrase_scoring(final_df, word_scores_df)
            
            # Tampilkan Hasil
            for index, row in final_scored_df.iterrows():
                st.write("Top 15 Keyphrase Ekstraksi beserta Skornya:")
                sorted_keyphrases = sorted(row['Keyphrase_Scores'], key=lambda x: x[1], reverse=True)
                ranked_keyphrases = self.rank_keyphrases(sorted_keyphrases)
                
                for i, (keyphrase, score) in enumerate(ranked_keyphrases, start=1):
                    st.write(f"{i}. {keyphrase}: {score}")

if __name__ == "__main__":
    Main().run()
