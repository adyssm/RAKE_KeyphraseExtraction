import nltk
from nltk.tag import CRFTagger
from nltk.chunk import RegexpParser

class POSTaggingChunking:
    def __init__(self):
        self.ct = CRFTagger()
        self.ct.set_model_file('C:\\SKRIPSI - Copy\\post_tagger.model')  # Update with your model path

    def chunk_np(self, pos_tagged_tokens):
        # Update grammar to only include NN, NNP, or NND in noun phrases
        grammar = r"""
            NP: {<DT>?<JJ>*<NN|NNP|NND>+}  # NP chunk: determiner (optional), adjectives (optional), followed by NN, NNP, or NND
        """
        chunk_parser = RegexpParser(grammar)
        tree = chunk_parser.parse(pos_tagged_tokens)

        noun_phrases = []
        current_phrase = []

        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                # Filter for subtrees that contain only NN, NNP, NND in the phrase
                phrase_tags = [pos for word, pos in subtree.leaves()]
                if all(tag in ['NN', 'NNP', 'NND'] for tag in phrase_tags):
                    current_phrase.extend([word for word, pos in subtree.leaves()])
            else:
                # If encountering another type, store phrase if valid and reset
                if current_phrase and len(current_phrase) <= 3:
                    noun_phrases.append(" ".join(current_phrase))
                current_phrase = []  # Reset phrase

        # Save the last phrase if it meets conditions
        if current_phrase and len(current_phrase) <= 3:
            noun_phrases.append(" ".join(current_phrase))

        return noun_phrases

    def apply_np_chunking(self, df):
        # Assuming 'combined_pos_tagged' contains POS-tagged tokens
        df['noun_phrase_chunks'] = df['combined_pos_tagged'].apply(self.chunk_np)
        return df
