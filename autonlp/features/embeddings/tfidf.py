from ...features.embeddings.base_embedding import Base_Embedding
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.random_projection import SparseRandomProjection
import numpy as np
import os
import json
import pickle
from hyperopt import hp
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ...features.cleaner import reduce_text_data
from ...utils.logging import get_logger

logger = get_logger(__name__)


class Tfidf(Base_Embedding):
    """ Base_Embedding class with TF-IDF embedding method from sklearn """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        Base_Embedding.__init__(self, flags_parameters, column_text, dimension_embedding)
        self.name_model = 'tf-idf'
        self.tokenizer = None
        self.max_features = None
        self.maxlen = None
        self.method_embedding = None
        self.embedding_matrix = None

    def hyper_params(self, size_params='small'):

        self.parameters = dict()

        if self.dimension_embedding == 'doc_embedding':
            # TF-IDF embedding
            if self.flags_parameters.tfidf_binary:
                self.parameters['vect__tfidf__binary'] = hp.choice('vect__tfidf__binary', [True, False])
            else:
                self.parameters['vect__tfidf__binary'] = hp.choice('vect__tfidf__binary', [False])
            self.parameters['vect__tfidf__ngram_range'] = hp.choice('vect__tfidf__ngram_range',
                                                                    self.flags_parameters.tfidf_ngram_range)
            if self.flags_parameters.tfidf_stop_words:
                if self.flags_parameters.language_text == 'fr':
                    stopwords = fr_stop
                else:
                    stopwords = en_stop
                self.parameters['vect__tfidf__stop_words'] = hp.choice('vect__tfidf__stop_words',
                                                                       [None, list(stopwords)])
            else:
                self.parameters['vect__tfidf__stop_words'] = hp.choice('vect__tfidf__stop_words', [None])

        else:
            self.parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.tfidf_wde_learning_rate)
        return self.parameters

    def preprocessing_fit_transform(self, x, doc_spacy_data, method_embedding):
        """ Fit preprocessing and transform x or doc_spacy_data according to embedding method and dimension embedding
            1st step : (Optional) reduce documents with pos_tag and lemmatization + clean text
            2nd step:
                - document dimension embedding : no more preprocessing to do
                - word dimension embedding : tensorflow tokenization + get word matrix embedding with TF-IDF method
        Args:
            x (Dataframe) need to have column column_text
            doc_spacy_data (List[spacy object])  list of documents processed by nlp.pipe spacy
            method_embedding (tuple(List[str],Boolean)) :
                - method_embedding[0] : 'all' for no pos_tag else list of tags to keeps
                - method_embedding[1] : apply lemmatization
        Return:
            - document dimension embedding : x_preprocessed (List(str))
            - word dimension embedding : x_token (dict)
        """

        self.method_embedding = method_embedding
        keep_pos_tag = self.method_embedding[0]
        lemmatize = self.method_embedding[1]
        if isinstance(self.column_text, int) and self.column_text not in x.columns:
            col = self.column_text
        else:
            col = list(x.columns).index(self.column_text)
        if doc_spacy_data is not None:
            x_preprocessed = reduce_text_data(doc_spacy_data, keep_pos_tag, lemmatize)
        else:
            x_preprocessed = list(x.iloc[:, col])

        if self.dimension_embedding == 'doc_embedding':
            return x_preprocessed
        else:
            # Tokenization by tensorflow with vocab size = max_features
            if keep_pos_tag == 'all':
                if lemmatize == True:
                    tokenizer_name = 'tokenizer_lem'
                else:
                    tokenizer_name = 'tokenizer_ALL'
            else:
                if lemmatize == True:
                    tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag) + '_lem'
                else:
                    tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag)

            if self.max_features is None:
                self.max_features = self.flags_parameters.tfidf_wde_max_features
            if self.maxlen is None:
                self.maxlen = self.flags_parameters.tfidf_wde_maxlen

            if self.tokenizer is None:
                if self.apply_logs:
                    dir_tokenizer = os.path.join(self.flags_parameters.outdir, tokenizer_name + '.pickle')
                elif self.apply_mlflow:
                    dir_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, tokenizer_name + '.pickle')
                if os.path.exists(dir_tokenizer):
                    with open(dir_tokenizer, 'rb') as handle:
                        self.tokenizer = pickle.load(handle)
                    logger.info("Load Tensorflow Tokenizer from past tokenization in {}".format(dir_tokenizer))
                else:
                    self.tokenizer = Tokenizer(num_words=self.max_features, lower=True, oov_token="<unk>")
                    self.tokenizer.fit_on_texts(x_preprocessed)

                    # Save Tokenizer :
                    if self.apply_logs:
                        path_tokenizer = os.path.join(self.flags_parameters.outdir, tokenizer_name + '.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.apply_mlflow:
                        path_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, tokenizer_name + '.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            tok = self.tokenizer.texts_to_sequences(x_preprocessed)

            self.word_index = self.tokenizer.word_index
            self.vocab_idx_word = {idx: word for word, idx in self.tokenizer.word_index.items()}
            self.length_word_index = len(self.word_index)

            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')

            x_token = {"tok": tok}

            self.embedding_matrix = build_embedding_matrix_from_tfidf(self.word_index, x_preprocessed,
                                                                      self.flags_parameters.tfidf_wde_stop_words,
                                                                      self.flags_parameters.language_text,
                                                                      self.flags_parameters.tfidf_wde_binary,
                                                                      self.flags_parameters.tfidf_wde_vector_size,
                                                                      self.seed)
            self.embed_size = self.embedding_matrix.shape[1]

            return x_token

    def preprocessing_transform(self, x, doc_spacy_data):
        """ Transform x or doc_spacy data according to latest fit preprocessing
        Args:
            x (Dataframe) need to have column column_text
            doc_spacy_data (List[spacy object])  list of documents processed by nlp.pipe spacy
        Return:
            - document dimension embedding : x_preprocessed (List(str))
            - word dimension embedding : x_token (dict)
        """
        if doc_spacy_data is not None:
            x_preprocessed = reduce_text_data(doc_spacy_data, self.method_embedding[0], self.method_embedding[1])
        else:
            if isinstance(self.column_text, int) and self.column_text not in x.columns:
                col = self.column_text
            else:
                col = list(x.columns).index(self.column_text)
            x_preprocessed = list(x.iloc[:, col])
        if self.dimension_embedding == 'doc_embedding':
            return x_preprocessed
        else:
            tok = self.tokenizer.texts_to_sequences(x_preprocessed)
            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')
            x_token = {"tok": tok}
            return x_token

    def save_params(self, outdir_model):
        params_all = dict()

        params_all['name_embedding'] = "tf-idf"
        params_all['method_embedding'] = self.method_embedding
        params_all['dimension_embedding'] = self.dimension_embedding
        params_all['language_text'] = self.flags_parameters.language_text

        if self.dimension_embedding == 'word_embedding':
            params_all['maxlen'] = self.maxlen
            params_all['embed_size'] = self.embed_size
            params_all['max_features'] = self.max_features
            params_all['length_word_index'] = self.length_word_index

        self.params_all = {self.name_model: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters_embedding.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

        return params_all

    def load_params(self, params_all, outdir):
        self.method_embedding = params_all['method_embedding']
        self.dimension_embedding = params_all['dimension_embedding']

        if self.dimension_embedding == 'word_embedding':
            self.maxlen = params_all['maxlen']
            self.embed_size = params_all['embed_size']
            self.max_features = params_all['max_features']
            self.length_word_index = params_all['length_word_index']
            try:
                keep_pos_tag = self.method_embedding[0]
                lemmatize = self.method_embedding[1]
                if keep_pos_tag == 'all':
                    if lemmatize == True:
                        tokenizer_name = 'tokenizer_lem'
                    else:
                        tokenizer_name = 'tokenizer_ALL'
                else:
                    if lemmatize == True:
                        tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag) + '_lem'
                    else:
                        tokenizer_name = 'tokenizer' + '_' + "_".join(keep_pos_tag)
                with open(os.path.join(outdir, tokenizer_name + '.pickle'), 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
            except:
                logger.warning("tokenizer.pickle is not provided in {}".format(outdir))

    def model(self):
        if self.dimension_embedding == 'doc_embedding':
            tfidf_ngrams = TfidfVectorizer()

            vect = ColumnTransformer(transformers=[
                ('tfidf', tfidf_ngrams, self.column_text)
            ])
            return vect

        else:
            token = tf.keras.layers.Input(shape=(self.maxlen,), name="tok")
            inp = {"tok": token}

            # Embedding
            if self.embedding_matrix is not None:
                x = Embedding(self.length_word_index + 1, self.embed_size, weights=[self.embedding_matrix],
                              trainable=True)(token)
            else:
                x = Embedding(self.length_word_index + 1, self.embed_size, trainable=True)(token)
            return x, inp


#################
# Help function : Get TF-IDF pre-training-weight to get word embeddings
#################


def get_embedding_index_tfidf(corpus, stopwords=True, language_text='fr', binary=True, vector_size=200, seed=15):
    """ Create a word vector for each word of the TF-IDF matrix of the corpus
        vector is obtained from a matrix reduction (SparseRandomProjection) of TF-IDF matrix
    Args:
        corpus (List[str]) a list of documents
        stopwords (Boolean)
        language_text (str) 'en' or 'fr'
        binary (Boolean) params of TF-IDF matrix
        vector_size (int) dimension reduction
        seed (int) for reproducibility
    Return:
         word_vectors (dict(str:array)) a word vector for each word of TF-IDF matrix
    """
    if stopwords:
        if language_text == 'fr':
            stopwords = fr_stop
        else:
            stopwords = en_stop
    else:
        stopwords = None
    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 1), binary=binary)
    X = vectorizer.fit_transform(corpus)
    if X.shape[0] > vector_size:
        srp = SparseRandomProjection(n_components=vector_size, dense_output=True, random_state=seed)
        word_embedding = srp.fit_transform(X.T)
        return dict(zip(vectorizer.get_feature_names(), word_embedding))
    else:
        return dict(zip(vectorizer.get_feature_names(), X.T.toarray()))


def build_embedding_matrix_from_tfidf(word_index, corpus, stopwords, language_text, binary, vector_size, seed):
    """ Create a word vector for each word in dictionary word_index with TF-IDF embedding method
    Args:
        word_index (dict) dictionary word:index got from tensorflow tokenization
        corpus (List[str]) a list of documents
        stopwords (Boolean)
        language_text (str) 'en' or 'fr'
        binary (Boolean) params of TF-IDF matrix
        vector_size (int) dimension reduction
        seed (int) for reproducibility
    Return:
         embedding_matrix (array) matrix of word vectors
    """
    embedding_matrix = None
    embedding_index = get_embedding_index_tfidf(corpus, stopwords, language_text, binary, vector_size, seed)
    for word, i in tqdm(word_index.items(), disable=False):
        word = word.lower()
        try:
            embedding_vector = embedding_index[word]
            if embedding_matrix is None:
                embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector.shape[0]))
        except:
            embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix