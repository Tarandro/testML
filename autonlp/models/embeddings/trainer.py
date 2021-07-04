import json
import os
import joblib
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.random_projection import SparseRandomProjection

import logging
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Embedding:
    """ Get Embedding """

    def __init__(self, flags_parameters, embedding, dimension_embedding, column_text):
        """
        Args:
            flags_parameters : Instance of Flags class object
            embedding (:class: Base_Embedding) : A children class from Base_Embedding
            dimension_embedding (str) : 'word_embedding' or 'doc_embedding'
            column_text (int) : column number with texts

        From flags_parameters:
        objective (str) : 'binary' or 'multi-class' or 'regression'
        average_scoring (str) : 'micro', 'macro' or 'weighted'
        seed (int)
        apply_mlflow (Boolean)
        experiment_name (str)
        """
        self.flags_parameters = flags_parameters

        self.embedding = embedding(flags_parameters, column_text, dimension_embedding)
        self.dimension_embedding = dimension_embedding

        self.column_text = column_text
        self.average_scoring = flags_parameters.average_scoring
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.seed = flags_parameters.seed

        if self.apply_mlflow:
            import mlflow
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

    def save_params(self, outdir_model):
        """ Save all params as a json file needed to reuse the embedding
            + tensorflow tokenizer (pickle file) in outdir_model

        Args:
            outdir_model (str)
        """
        params_embedding = self.embedding.save_params(outdir_model)

        if self.embedding.name_model in ["tf","tf-idf"]:
            _ = joblib.dump(self.vec, os.path.join(outdir_model,"vectorizer.joblib.pkl"))

            if self.srp is not None:
                _ = joblib.dump(self.srp, os.path.join(outdir_model, "projection.joblib.pkl"))

    def load_params(self, params_all, outdir):
        """ Initialize all params from params_all
            + tensorflow tokenizer (pickle file) from outdir path

        Args:
            params_all (dict)
            outdir (str)
        """
        self.embedding.load_params(params_all, outdir)

        if self.embedding.name_model in ["tf", "tf-idf"]:
            outdir_model = os.path.join(self.flags_parameters.outdir, 'embedding', self.embedding.name_model)
            self.vec = joblib.load(os.path.join(outdir_model, "vectorizer.joblib.pkl"))

            try:
                self.srp = joblib.load(os.path.join(outdir_model, "projection.joblib.pkl"))
            except:
                self.srp = None

    def fit_transform_vectorization_tf_tfidf(self, x_train_preprocessed):

        if self.embedding.name_model == "tf":
            stopwords = self.flags_parameters.tf_wde_stop_words
            binary = self.flags_parameters.tf_wde_binary
            self.vector_size = self.flags_parameters.tf_wde_vector_size
            ngram_range = self.flags_parameters.tf_wde_ngram_range
            Vectorizer = CountVectorizer
        elif self.embedding.name_model == "tf-idf":
            stopwords = self.flags_parameters.tfidf_wde_stop_words
            binary = self.flags_parameters.tfidf_wde_binary
            self.vector_size = self.flags_parameters.tfidf_wde_vector_size
            ngram_range = self.flags_parameters.tfidf_wde_ngram_range
            Vectorizer = TfidfVectorizer
        else:
            return x_train_preprocessed

        if stopwords:
            if self.flags_parameters.language_text == 'fr':
                stopwords = fr_stop
            else:
                stopwords = en_stop
        else:
            stopwords = None
        self.vec = Vectorizer(stop_words=stopwords, ngram_range=ngram_range, binary=binary)
        doc_vectors = self.vec.fit_transform(x_train_preprocessed)
        if doc_vectors.shape[0] > self.vector_size:
            self.srp = SparseRandomProjection(n_components=self.vector_size, dense_output=True, random_state=self.seed)
            doc_vectors = self.srp.fit_transform(doc_vectors)
        else:
            self.srp = None
        return doc_vectors

    def transform_vectorization_tf_tfidf(self, x_preprocessed):

        doc_vectors = self.vec.transform(x_preprocessed)
        if self.srp is not None:
            doc_vectors = self.srp.transform(doc_vectors)
        return doc_vectors

    def fit_transform(self, x_train_before, x_val_before=None, method_embedding=None,
                      doc_spacy_data_train=[], doc_spacy_data_val=[]):
        """ Apply fit_transform from :class:embedding on x_train_before and transform on x_val_before
        Args:
            x_train_before (Dataframe)
            x_val_before (Dataframe)
            method_embedding (str) name of embedding method or path of embedding weights
            doc_spacy_data_train (List[spacy object])
            doc_spacy_data_val (List[spacy object])
        """

        x_train = x_train_before.copy()
        if x_val_before is not None:
            x_val = x_val_before.copy()
        else:
            x_val = None

        self.method_embedding = method_embedding

        # preprocess text on x_train :
        if self.embedding.name_model not in ['tf', 'tf-idf']:
            x_train_preprocessed = self.embedding.preprocessing_fit_transform(x_train, self.flags_parameters.size_params,
                                                                              self.method_embedding)
            if x_val is not None:
                x_val_preprocessed = self.embedding.preprocessing_transform(x_val)
        else:
            x_train_preprocessed = self.embedding.preprocessing_fit_transform(x_train, doc_spacy_data_train,
                                                                              self.method_embedding)
            x_train_preprocessed = self.fit_transform_vectorization_tf_tfidf(x_train_preprocessed)
            if x_val is not None:
                x_val_preprocessed = self.embedding.preprocessing_transform(x_val, doc_spacy_data_val)
                x_val_preprocessed = self.transform_vectorization_tf_tfidf(x_val_preprocessed)

        # save params in path : 'outdir/last_logs/name_embedding/name_model_full'
        if self.apply_logs:
            outdir_file_embedding = os.path.join(self.flags_parameters.outdir, 'embedding')
            os.makedirs(outdir_file_embedding, exist_ok=True)
            outdir_embedding = os.path.join(outdir_file_embedding, self.embedding.name_model)
            os.makedirs(outdir_embedding, exist_ok=True)
            self.save_params(outdir_embedding)
        else:
            self.save_params(None)

        dict_preprocessed = {"x_train_preprocessed": x_train_preprocessed}
        if x_val is not None:
            dict_preprocessed["x_val_preprocessed"] = x_val_preprocessed
        if self.dimension_embedding == 'word_embedding':
            try:  # no embedding matrix for Transformer model (load on internet)
                dict_preprocessed["embedding_matrix"] = self.embedding.embedding_matrix
            except:
                pass
        return dict_preprocessed

    def transform(self, x_test_before_copy, doc_spacy_data_test=[]):
        """ Apply transform from :class:embedding on x_test_before_copy
        Args:
            x_test_before_copy (List or dict or DataFrame)
            doc_spacy_data_test (List[spacy object])
        """
        x_test = x_test_before_copy.copy()

        outdir_file_embedding = os.path.join(self.flags_parameters.outdir, 'embedding')
        outdir_embedding = os.path.join(outdir_file_embedding, self.embedding.name_model)
        with open(os.path.join(outdir_embedding, "parameters_embedding.json")) as json_file:
            params_all = json.load(json_file)
        params_all = params_all[self.embedding.name_model]
        self.load_params(params_all, os.path.join(self.flags_parameters.outdir))

        if self.embedding.name_model not in ['tf', 'tf-idf']:
            x_test_preprocessed = self.embedding.preprocessing_transform(x_test)
        else:
            x_test_preprocessed = self.embedding.preprocessing_transform(x_test, doc_spacy_data_test)
            x_test_preprocessed = self.transform_vectorization_tf_tfidf(x_test_preprocessed)
        return x_test_preprocessed


