import os
import pandas as pd
import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.doc2vec import TaggedDocument
from gensim.models import KeyedVectors

import re
import string


def small_clean_text(text):
    """ Clean text : lower text + Remove '\n', '\r', URL, '’', numbers and double space
    Args:
        text (str)
    Return:
        text (str)
    """
    text = str(text).lower()

    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('’', ' ', text)
    text = re.sub('’', ' ', text)
    text = re.sub('«', ' ', text)
    text = re.sub('“', ' ', text)
    text = re.sub('”', ' ', text)
    text = re.sub('»', ' ', text)
    text = re.sub('…', ' ', text)
    text = re.sub('‟', ' ', text)

    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)

    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002500-\U00002BEF"  # chinese char
                                        u"\U00002702-\U000027B0"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u"\U00010000-\U0010ffff"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\x07"
                                        u"\x08"
                                        u"\ufe0f"  # dingbats
                                        u"\u3030"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    text = regrex_pattern.sub(r'', text)

    return text


def extract_sentences(text):
    """ Extract sentences by splitting text by '!' or '?' or '.'
    Args:
        text (str)
    Return:
        sentences (List[str])
    """
    sentences = text.replace('!', '.').replace('?', '.').split(". ")
    return sentences


def clean_sentence(sentence):
    """ Clean sentence : remove Punctuation and double space
    Args:
        sentence (str)
    Return:
        sentence (str)
    """
    sentence = re.sub('[%s]' % re.escape(string.punctuation), ' ', sentence)  # remove punctuation
    sentence = re.sub(' +', ' ', sentence)
    return sentence


def get_clean_sentences(data):
    """ For each document, apply small_clean_text, extraction of sentences and clean_sentence
    Args:
        data (List[str])
    Return:
        list_sentences_clean (List[str])
    """
    data = [small_clean_text(text) for text in data]

    list_sentences = []
    for text in data:
        list_sentences.extend(extract_sentences(text))

    list_sentences_clean = [clean_sentence(sentence) for sentence in list_sentences]
    return list_sentences_clean


def split_and_clean_sentences(data_sentences):
    """ Split sentences in words and remove word with length = 1
    Args:
        data_sentences (List[str])
    Return:
        list_sentences_clean (List[List[str]])
    """
    data_split = []
    for sentence in data_sentences:
        sentence_split = sentence.split()
        sentence_split = [word for word in sentence_split if len(word) > 1]
        if len(sentence_split) > 1:
            data_split.append(sentence_split)
    return data_split


def build_word2vec_model(data_not_preprocessed, output_dir=".", size_vector=300, window=5, epochs=30, sg=0):
    """ Build a Word2Vec model and save word vectors
    Args:
        data_not_preprocessed (Series)
        output_dir (str) directory to save word vector
        size_vector (int) Dimensionality of the word vectors
        window (int) Maximum distance between the current and predicted word within a sentence
        epochs (int) Number of iterations over the corpus
        sg ({0, 1}) Training algorithm: 1 for skip-gram; otherwise CBOW.
    """
    data = data_not_preprocessed.copy()
    # Texts in a single list of word lists split by sentence:
    data_sentences = get_clean_sentences(data)
    data_split = split_and_clean_sentences(data_sentences)

    # Build Word2vec model
    if gensim.__version__[0] == '4':
        model = gensim.models.Word2Vec(vector_size=size_vector, workers=2, window=window, min_count=1, sg=sg)
        model.build_vocab(corpus_iterable=data_split)
        model.train(corpus_iterable=data_split, total_examples=len(data_split), epochs=epochs)
    else:
        model = gensim.models.Word2Vec(size=size_vector, workers=2, window=window, min_count=1, sg=sg)
        model.build_vocab(sentences=data_split)
        model.train(sentences=data_split, total_examples=len(data_split), epochs=epochs)

    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    os.makedirs(output_dir, exist_ok=True)
    word_vectors.save(os.path.join(output_dir, "word2vec.wordvectors"))


def build_fasttext_model(data_not_preprocessed, output_dir=".", thr_grams=10, size_vector=300, window=5, epochs=30, sg=0):
    """ Build a FastText model and save word vectors
    Args:
        data_not_preprocessed (Series)
        output_dir (str) directory to save word vector
        thr_grams (int) Represent a score threshold for forming the phrases (higher means fewer phrases)
        size_vector (int) Dimensionality of the word vectors
        window (int) Maximum distance between the current and predicted word within a sentence
        epochs (int) Number of iterations over the corpus
        sg ({0, 1}) Training algorithm: 1 for skip-gram; otherwise CBOW.
    """
    data = data_not_preprocessed.copy()
    # Texts in a single list of word lists split by sentence:
    data_sentences = get_clean_sentences(data)
    data_split = split_and_clean_sentences(data_sentences)
    # Create the relevant phrases from the list of sentences:
    bigram_phrases = Phrases(data_split, threshold=thr_grams)  # default threshold = 10
    # The Phraser object is used from now on to transform sentences
    bigram = Phraser(bigram_phrases)
    trigram_phrases = Phrases(bigram[data_split], threshold=thr_grams)
    trigram = Phraser(trigram_phrases)
    # Applying the Phraser to transform our sentences is simply
    data_split = list(trigram[bigram[data_split]])

    # Build FastText model
    if gensim.__version__[0] == '4':
        model = gensim.models.FastText(vector_size=size_vector, workers=2, window=window, min_count=1, sg=sg)
        model.build_vocab(corpus_iterable=data_split)
        model.train(corpus_iterable=data_split, total_examples=len(data_split), epochs=epochs)
    else:
        model = gensim.models.FastText(size=size_vector, workers=2, window=window, min_count=1, sg=sg)
        model.build_vocab(sentences=data_split)
        model.train(sentences=data_split, total_examples=len(data_split), epochs=epochs)

    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    os.makedirs(output_dir, exist_ok=True)
    word_vectors.save(os.path.join(output_dir, "fasttext.wordvectors"))


def build_doc2vec_model(data_not_preprocessed, output_dir=".", size_vector=300, window=5, epochs=30, dm=0):
    """ Build a Doc2Vec model and save word vectors
    Args:
        data_not_preprocessed (Series)
        output_dir (str) directory to save word vector
        size_vector (int) Dimensionality of the word vectors
        window (int) Maximum distance between the current and predicted word within a sentence
        epochs (int) Number of iterations over the corpus
        dm ({0, 1}) Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used.
                    Otherwise, distributed bag of words (PV-DBOW) is employed
    """
    data = data_not_preprocessed.copy()
    # Texts in a single list of word lists split by sentence:
    data_sentences = get_clean_sentences(data)
    data_split = split_and_clean_sentences(data_sentences)

    train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(data_split)]

    # Build FastText model
    if gensim.__version__[0] == '4':
        model = gensim.models.doc2vec.Doc2Vec(train_corpus, dm=dm, vector_size=size_vector, window=window, min_count=1, workers=2,
                                              dbow_words=1)
        #model.build_vocab(corpus_iterable=train_corpus)
        #model.train(corpus_iterable=train_corpus, total_examples=len(train_corpus), epochs=epochs)
    else:
        model = gensim.models.doc2vec.Doc2Vec(dm=dm, vector_size=size_vector, window=window, min_count=1, workers=2,
                                              dbow_words=1)
        model.build_vocab(documents=train_corpus)
        model.train(documents=train_corpus, total_examples=len(train_corpus), epochs=epochs)

    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    os.makedirs(output_dir, exist_ok=True)
    word_vectors.save(os.path.join(output_dir, "doc2vec.wordvectors"))


path_data = "../../data/FinancialPhraseBank.csv"
column_text = "text_fr"

name_embedding = "doc2vec"

size_vector = 300
window = 5
epochs = 50
thr_grams = 10
sg = 0
dm = 0

output_dir = "../../.."

if __name__ == '__main__':

    data = pd.read_csv(path_data)

    if name_embedding.lower() == "fasttext":
        build_fasttext_model(data[column_text], output_dir=output_dir, thr_grams=thr_grams, size_vector=size_vector,
                             window=window, epochs=epochs, sg=sg)

    elif name_embedding.lower() == "word2vec":
        build_word2vec_model(data[column_text], output_dir=output_dir, size_vector=size_vector, window=window,
                             epochs=epochs, sg=sg)

    elif name_embedding.lower() == "doc2vec":
        build_doc2vec_model(data[column_text], output_dir=output_dir, size_vector=size_vector, window=window,
                             epochs=epochs, dm=dm)


    wv = KeyedVectors.load(os.path.join(output_dir, name_embedding.lower()+".wordvectors"), mmap='r')
    vector = wv['usine']  # Get numpy vector of a word
    print('usine')
    print(wv.most_similar('usine'))

    print('euros')
    print(wv.most_similar('euros'))

    print('entreprise')
    print(wv.most_similar('entreprise'))