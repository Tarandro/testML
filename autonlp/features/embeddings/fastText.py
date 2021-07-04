from ...features.embeddings.base_embedding import Base_Embedding
import numpy as np
from tqdm import tqdm
import pickle
import os
import json
import gensim
import re
import string
from hyperopt import hp
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from ...features.embeddings.gensim_model.scratch_gensim_model import build_fasttext_model

from ...utils.logging import get_logger

logger = get_logger(__name__)


class Fasttext(Base_Embedding):
    """ Base_Embedding class with Fasttext embedding method from gensim or pre-trained Fasttext weights """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        Base_Embedding.__init__(self, flags_parameters, column_text, dimension_embedding)
        self.name_model = 'fasttext'
        self.tokenizer = None
        self.embed_size = None
        self.max_features = None
        self.maxlen = None
        self.method_embedding = None
        self.embedding_matrix = None

    def hyper_params(self, size_params='small'):

        self.parameters = dict()

        self.parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.ft_learning_rate)
        return self.parameters

    def init_params(self, size_params='small', method_embedding='fasttext'):
        if self.dimension_embedding == 'word_embedding':
            if size_params == 'small':
                if self.max_features is None:
                    self.max_features = self.flags_parameters.ft_max_features
                if self.maxlen is None:
                    self.maxlen = self.flags_parameters.ft_maxlen
            else:
                if self.max_features is None:
                    self.max_features = self.flags_parameters.ft_max_features  # 100000
                if self.maxlen is None:
                    self.maxlen = self.flags_parameters.ft_maxlen  # 350
        self.method_embedding = method_embedding

    def preprocessing_fit_transform(self, x, size_params='small', method_embedding='fasttext'):
        """ Fit preprocessing and transform x according to embedding method and dimension embedding
            1st step : initialize some fixed params needed for embedding method
            2nd step : Build a FastText scratch model or use a pre-trained FastText model/weights
            3rd step:
                - word dimension embedding : tensorflow tokenization + get word matrix embedding with FastText method
                - document dimension embedding : get document vectors with FastText method
        Args:
            x (Dataframe) need to have column column_text
            size_params ('small' or 'big') size of parameters range for optimization
            method_embedding (str) 'fasttext' if want to use a scratch model else a path for a pre-trained model/weights
        Return:
            - word dimension embedding : x_preprocessed (dict)
            - document dimension embedding : document_embedding (array) a matrix of document vectors
        """
        self.init_params(size_params, method_embedding)

        if isinstance(self.column_text, int) and self.column_text not in x.columns:
            col = self.column_text
        else:
            col = list(x.columns).index(self.column_text)

        # build gensim scratch model
        if self.method_embedding.lower() == 'fasttext':
            if self.apply_logs:
                dir_fasttext = os.path.join(self.flags_parameters.outdir, 'FastText')
            elif self.apply_mlflow:
                dir_fasttext = os.path.join(self.path_mlflow, self.experiment_id, 'FastText')
            if not os.path.exists(dir_fasttext):
                os.makedirs(dir_fasttext, exist_ok=True)
                logger.info(
                    "Build FastText model from scratch with train set and size_vector={}, window={}, epochs={}, thr_grams={} ...".format(
                        self.flags_parameters.ft_size_vector, self.flags_parameters.ft_window,
                        self.flags_parameters.ft_epochs, self.flags_parameters.ft_thr_grams))
                build_fasttext_model(list(x.iloc[:, col]), output_dir=dir_fasttext,
                                     thr_grams=self.flags_parameters.ft_thr_grams,
                                     size_vector=self.flags_parameters.ft_size_vector,
                                     window=self.flags_parameters.ft_window, epochs=self.flags_parameters.ft_epochs)
                logger.info("Save FastText model in '{}'".format(dir_fasttext))
                self.method_embedding = os.path.join(dir_fasttext, "fasttext.wordvectors")
            else:
                self.method_embedding = os.path.join(dir_fasttext, "fasttext.wordvectors")
                logger.info("Load FastText scratch model from path : {}".format(dir_fasttext))

        # build_embedding_matrix(self):
        try:
            try:
                embeddings_index = load_embeddings(self.method_embedding)
            except:
                embeddings_index = load_keyedvectors(self.method_embedding)
        except Exception:
            logger.critical("unknown path for FastText weights : '{}'".format(self.method_embedding))

        if self.dimension_embedding == 'word_embedding':
            # Tokenization by tensorflow with vocab size = max_features
            if self.tokenizer is None:
                if self.apply_logs:
                    dir_tokenizer = os.path.join(self.flags_parameters.outdir, 'tokenizer.pickle')
                elif self.apply_mlflow:
                    dir_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, 'tokenizer.pickle')
                if os.path.exists(dir_tokenizer):
                    with open(dir_tokenizer, 'rb') as handle:
                        self.tokenizer = pickle.load(handle)
                    logger.info("Load Tensorflow Tokenizer from past tokenization in {}".format(dir_tokenizer))
                else:
                    self.tokenizer = Tokenizer(num_words=self.max_features, lower=True, oov_token="<unk>")
                    self.tokenizer.fit_on_texts(list(x.iloc[:, col]))

                    # Save Tokenizer :
                    if self.apply_logs:
                        path_tokenizer = os.path.join(self.flags_parameters.outdir, 'tokenizer.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.apply_mlflow:
                        path_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, 'tokenizer.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            tok = self.tokenizer.texts_to_sequences(x.iloc[:, col])

            self.word_index = self.tokenizer.word_index
            self.vocab_idx_word = {idx: word for word, idx in self.tokenizer.word_index.items()}
            self.length_word_index = len(self.word_index)

            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')

            x_preprocessed = {"tok": tok}

            self.embedding_matrix = build_embedding_matrix_from_embeddings_index(self.word_index, embeddings_index)
            self.embed_size = self.embedding_matrix.shape[1]

            return x_preprocessed

        else:
            document_embedding = build_embedding_documents_from_embeddings_index(list(x.iloc[:, col]),
                                                                                 embeddings_index)
            return document_embedding

    def preprocessing_transform(self, x):
        """ Transform x data according to latest fit preprocessing
        Args:
            x (Dataframe) need to have column column_text
        Return:
            - word dimension embedding : x_preprocessed (dict)
            - document dimension embedding : document_embedding (array) a matrix of document vectors
        """
        if isinstance(self.column_text, int) and self.column_text not in x.columns:
            col = self.column_text
        else:
            col = list(x.columns).index(self.column_text)

        if self.dimension_embedding == 'word_embedding':
            tok = self.tokenizer.texts_to_sequences(x.iloc[:, col])
            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')
            x_preprocessed = {"tok": tok}
            return x_preprocessed
        else:
            try:
                try:
                    embeddings_index = load_embeddings(self.method_embedding)
                except:
                    embeddings_index = load_keyedvectors(self.method_embedding)
            except Exception:
                logger.critical("unknown path for FastText weights : '{}'".format(self.method_embedding))

            document_embedding = build_embedding_documents_from_embeddings_index(x.iloc[:, col],
                                                                                 embeddings_index)
            return document_embedding

    def save_params(self, outdir_model):
        params_all = dict()
        params_all['name_embedding'] = "fasttext"
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
            with open(os.path.join(outdir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except:
            logger.warning("tokenizer.pickle is not provided in {}".format(outdir))

        if self.method_embedding.lower() == 'fasttext' or not os.path.exists(self.method_embedding):
            dir_fasttext = os.path.join(outdir, 'FastText')
            if os.path.exists(dir_fasttext):
                self.method_embedding = os.path.join(dir_fasttext, "fasttext.wordvectors")
            else:
                logger.error("A directory 'FastText' with the model must be provided in {}".format(outdir))

    def model(self):

        if self.dimension_embedding == 'word_embedding':
            token = tf.keras.layers.Input(shape=(self.maxlen,), name="tok")
            inp = {"tok": token}

            # Embedding
            if self.embedding_matrix is not None:
                x = Embedding(self.length_word_index + 1, self.embed_size, weights=[self.embedding_matrix],
                              trainable=True, input_shape=(self.length_word_index + 1, ))(token)
            else:
                x = Embedding(self.length_word_index + 1, self.embed_size, trainable=True,
                              input_shape=(self.length_word_index + 1, ))(token)
            return x, inp

#################
# Help function : Get fasttext pre-training-weight and attention head
#################

def load_model(embed_dir):
    """ Load a full gensim model
    Args:
        embed_dir (str) path of gensim model
        model are often separated in several files but it only need the path of one file
        all files need to be in the same directory
    Return:
        model (gensim model)
    """
    # need to have gensim model + syn0.npy + syn1neg.npy
    model = gensim.models.FastText.load(embed_dir)
    return model


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(embed_dir):
    """ read a .vec file and get all word vectors of each word in the file
    Args:
        embed_dir (str) path of pre-trained word vectors
    Return:
        embedding_index (dict)
    """
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir, 'rt', encoding='utf-8'))
    return embedding_index


def load_keyedvectors(embed_dir):
    """ Load a word vector gensim model : the model have only the option to give vector of a string
    Args:
        embed_dir (str) path of word vector gensim model
        model are often separated in several files but it only need the path of one file
        all files need to be in the same directory
    Return:
        embedding_index (word vector gensim model)
    """
    # need to have file : fasttext.wordvectors + fasttext.wordvectors.vectors_ngrams.npy
    embedding_index = gensim.models.KeyedVectors.load(embed_dir)
    return embedding_index


def get_vect(word, embeddings_index):
    """ Obtain the vector of a word with embeddings_index
    Args:
        word (str)
        embeddings_index (dict)
    Return:
        vector (array)
    """
    try:
        return embeddings_index[word]
    except KeyError:
        return None


def build_embedding_matrix_from_embeddings_index(word_index, embeddings_index, lower=True, verbose=True):
    """ Create a word vector for each word in dictionary word_index with a embeddings index
    Args:
        word_index (dict) dictionary word:index got from tensorflow tokenization
        embeddings_index (dict) word vectors
        lower (Boolean) lower each word of embedding matrix
        verbose (Boolean) show iteration progression in word_index
    Return:
         embedding_matrix (array) matrix of word vectors
    """
    embedding_matrix = None
    for word, i in tqdm(word_index.items(), disable=not verbose):
        if lower:
            word = word.lower()
        embedding_vector = get_vect(word, embeddings_index)
        if embedding_matrix is None and embedding_vector is not None:
            embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector.shape[0]))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_embedding_documents_from_embeddings_index(documents, embeddings_index, lower=True, verbose=True):
    """ Create a document vector for each document in documents with a embeddings index
        and concatenate to get an embedding matrix
    Args:
        documents (List[str])
        embeddings_index (dict) word vectors
        lower (Boolean) lower each word of embedding matrix
        verbose (Boolean) show iteration progression in word_index
    Return:
        embedding_documents (array) matrix of document vectors
    """
    embedding_documents = None
    for i, doc in tqdm(enumerate(documents), disable=not verbose):
        if lower:
            doc = doc.lower()
        doc = re.sub('[%s]' % re.escape(string.punctuation), ' ', doc)
        try:
            doc_split = doc.split(' ')
            embedding_vector = [get_vect(word, embeddings_index) for word in doc_split]
            embedding_vector = [i for i in embedding_vector if i is not None]
            if len(embedding_vector) < 1:
                embedding_vector = None
            else:
                embedding_vector = sum(embedding_vector)
            if embedding_documents is None and embedding_vector is not None:
                embedding_documents = np.zeros((len(documents), embedding_vector.shape[0]))
        except:
            embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_documents[i] = embedding_vector
    return embedding_documents
