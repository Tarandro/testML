from ...features.embeddings.base_embedding import Base_Embedding
import numpy as np
import os
import json
from hyperopt import hp
import tensorflow as tf

import transformers
import tokenizers

from ...utils.logging import get_logger

logger = get_logger(__name__)


class TransformerNLP(Base_Embedding):
    """ Base_Embedding class with TransformerNLP embedding method from Huggingface library """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        Base_Embedding.__init__(self, flags_parameters, column_text, dimension_embedding)
        self.name_model = 'transformer'
        self.maxlen = None
        self.tokenizer = None

    def hyper_params(self, size_params='small'):

        self.parameters = dict()
        self.parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.tr_learning_rate)
        return self.parameters

    def init_params(self, size_params='small', method_embedding='CamemBERT'):
        if self.maxlen is None:
            if size_params == 'small':
                self.maxlen = self.flags_parameters.tr_maxlen
            else:
                self.maxlen = self.flags_parameters.tr_maxlen  # 350

        language = self.flags_parameters.language_text
        self.method_embedding = method_embedding
        # https://huggingface.co/models?filter=tf
        models_huggingface_fr = ['CamemBERT', 'FlauBERT']
        models_huggingface_en = ['RoBERTa', 'BERT']
        models_huggingface_multilingual = ['XLM-RoBERTa']

        if method_embedding not in models_huggingface_fr + models_huggingface_en + models_huggingface_multilingual:
            logger.critical("unknown embedding method name : '{}'".format(method_embedding))
        elif method_embedding in models_huggingface_fr and language != 'fr':
            logger.warning("'{}' is a 'fr' language model but you set language = {}".format(method_embedding, language))
        elif method_embedding in models_huggingface_en and language != 'en':
            logger.warning("'{}' is a 'en' language model but you set language = {}".format(method_embedding, language))

        if self.tokenizer is None:
            # Instantiate tokenizer
            if method_embedding == 'RoBERTa':
                # experience :
                PATH = '/kaggle/input/tf-roberta/'
                self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                    vocab=PATH + 'vocab-roberta-base.json',
                    merges=PATH + 'merges-roberta-base.txt',
                    lowercase=True,
                    add_prefix_space=True
                )
                # sinon
                # self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
            elif method_embedding == 'BERT':
                self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            elif method_embedding == 'CamemBERT':
                self.tokenizer = transformers.CamembertTokenizer.from_pretrained('jplu/tf-camembert-base')
            elif method_embedding == 'FlauBERT':
                self.tokenizer = transformers.FlaubertTokenizer.from_pretrained('jplu/tf-flaubert-base-uncased')
            elif method_embedding == 'XLM-RoBERTa':
                self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('jplu/tf-xlm-roberta-base')
            else:
                logger.critical("unknown embedding method name : '{}'".format(method_embedding))

    def preprocessing_fit_transform(self, x, size_params='small', method_embedding='CamemBERT'):
        """ Fit preprocessing and transform x according to embedding method and dimension embedding
            1st step : initialize some fixed params and tokenizer needed for embedding method
            2nd step : Transformer Tokenization
            3rd step:
                - word dimension embedding : no more preprocessing to do
                - document dimension embedding : get document vectors with Transformer pre-trained model
        Args:
            x (Dataframe) need to have column column_text
            size_params ('small' or 'big') size of parameters range for optimization
            method_embedding (str) name of a Transformer model
        Return:
            - word dimension embedding : x_preprocessed (dict)
            - document dimension embedding : document_embedding (array) a matrix of document vectors
        """
        self.init_params(size_params, method_embedding)

        if isinstance(self.column_text, int) and self.column_text not in x.columns:
            col = self.column_text
        else:
            col = list(x.columns).index(self.column_text)

        ct = x.shape[0]
        # INPUTS
        if self.method_embedding.lower() in ['roberta', "camembert", "xlm-roberta"]:
            ids = np.ones((ct, self.maxlen), dtype='int32')
        else:
            ids = np.zeros((ct, self.maxlen), dtype='int32')
        att = np.zeros((ct, self.maxlen), dtype='int32')
        tok = np.zeros((ct, self.maxlen), dtype='int32')

        for k in range(ct):
            text = "  " + " ".join(x.iloc[k, col].split())

            if self.method_embedding == 'RoBERTa':
                enc = self.tokenizer.encode(text)
            else:
                enc = self.tokenizer.encode(text, max_length=self.maxlen, truncation=True)

            # CREATE BERT INPUTS
            if self.method_embedding == 'RoBERTa':
                ids[k, :len(enc.ids)] = enc.ids[:self.maxlen]
                att[k, :len(enc.ids)] = 1
            else:
                ids[k, :len(enc)] = enc
                att[k, :len(enc)] = 1

        x_preprocessed = [ids, att, tok]
        if self.dimension_embedding == 'word_embedding':
            return x_preprocessed
        else:
            model_extractor = self.model_extract_document_embedding()
            document_embedding = model_extractor.predict(x_preprocessed)
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

        ct = x.shape[0]
        # INPUTS
        if self.method_embedding.lower() in ['roberta', "camembert", "xlm-roberta"]:
            ids = np.ones((ct, self.maxlen), dtype='int32')
        else:
            ids = np.zeros((ct, self.maxlen), dtype='int32')
        att = np.zeros((ct, self.maxlen), dtype='int32')
        tok = np.zeros((ct, self.maxlen), dtype='int32')

        for k in range(ct):
            text = "  " + " ".join(x.iloc[k, col].split())

            if self.method_embedding == 'RoBERTa':
                enc = self.tokenizer.encode(text)
            else:
                enc = self.tokenizer.encode(text, max_length=self.maxlen, truncation=True)

            # CREATE BERT INPUTS
            if self.method_embedding == 'RoBERTa':
                ids[k, :len(enc.ids)] = enc.ids[:self.maxlen]
                att[k, :len(enc.ids)] = 1
            else:
                ids[k, :len(enc)] = enc
                att[k, :len(enc)] = 1

        x_preprocessed = [ids, att, tok]
        if self.dimension_embedding == 'word_embedding':
            return x_preprocessed
        else:
            model_extractor = self.model_extract_document_embedding()
            document_embedding = model_extractor.predict(x_preprocessed)
            return document_embedding

    def model_extract_document_embedding(self):
        """ Create a Tensorflow model which extract [CLS] token output of the Transformer model
            [CLS] token output can be used as a document vector
        Return:
            model (Tensorflow model)
        """
        input_ids = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="ids")
        attention_mask = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="att")
        token = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="tok")

        # Embedding :
        if self.method_embedding == 'CamemBERT':
            Camembert_model = transformers.TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
            x = Camembert_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'FlauBERT':
            # lr = 0.00001
            Flaubert_model = transformers.TFFlaubertModel.from_pretrained("jplu/tf-flaubert-base-uncased")
            x = Flaubert_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'XLM-RoBERTa':
            # lr = 0.00001
            XLMRoBERTa_model = transformers.TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")
            x = XLMRoBERTa_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'RoBERTa':
            # Experience Test path weights :
            PATH = '/kaggle/input/tf-roberta/'
            config = transformers.RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
            Roberta_model = transformers.TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5',
                                                                        config=config)
            # Sinon :
            # Roberta_model = transformers.TFRobertaModel.from_pretrained('roberta-base')
            x = Roberta_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'BERT':
            BERT_model = transformers.TFBertModel.from_pretrained('bert-base-uncased')
            x = BERT_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        else:
            logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))

        # word vectors shape : (None, maxlen, 768)
        x = x[0]
        cls_token = x[:, 0, :]

        model = tf.keras.models.Model(inputs=[input_ids, attention_mask, token], outputs=cls_token)
        return model

    def save_params(self, outdir_model):
        params_all = dict()
        params_all['name_embedding'] = "transformer"
        params_all['maxlen'] = self.maxlen
        params_all['method_embedding'] = self.method_embedding
        params_all['dimension_embedding'] = self.dimension_embedding
        params_all['language_text'] = self.flags_parameters.language_text

        self.params_all = {self.name_model: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters_embedding.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

        return params_all

    def load_params(self, params_all, outdir):
        self.maxlen = params_all['maxlen']
        self.method_embedding = params_all['method_embedding']
        self.dimension_embedding = params_all['dimension_embedding']
        self.maxlen = params_all['maxlen']

        # Instantiate tokenizer
        if self.method_embedding == 'RoBERTa':
            # experience, use a RoBERTa save in a file:
            PATH = '/kaggle/input/tf-roberta/'
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab=PATH + 'vocab-roberta-base.json',
                merges=PATH + 'merges-roberta-base.txt',
                lowercase=True,
                add_prefix_space=True
            )
            # else
            # self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
        elif self.method_embedding == 'BERT':
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.method_embedding == 'CamemBERT':
            self.tokenizer = transformers.CamembertTokenizer.from_pretrained('jplu/tf-camembert-base')
        elif self.method_embedding == 'FlauBERT':
            self.tokenizer = transformers.FlaubertTokenizer.from_pretrained('jplu/tf-flaubert-base-uncased')
        elif self.method_embedding == 'XLM-RoBERTa':
            self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('jplu/tf-xlm-roberta-base')
        else:
            logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))

    def model(self):
        input_ids = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="ids")
        attention_mask = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="att")
        token = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="tok")

        inp = [input_ids, attention_mask, token]

        # Embedding + GlobalAveragePooling1D
        if self.method_embedding == 'CamemBERT':
            Camembert_model = transformers.TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
            x = Camembert_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'FlauBERT':
            # lr = 0.00001
            Flaubert_model = transformers.TFFlaubertModel.from_pretrained("jplu/tf-flaubert-base-uncased")
            x = Flaubert_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'XLM-RoBERTa':
            # lr = 0.00001
            XLMRoBERTa_model = transformers.TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")
            x = XLMRoBERTa_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'RoBERTa':
            # Experience Test path weights :
            PATH = '/kaggle/input/tf-roberta/'
            config = transformers.RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
            Roberta_model = transformers.TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5',
                                                                        config=config)
            # Sinon :
            # Roberta_model = transformers.TFRobertaModel.from_pretrained('roberta-base')
            x = Roberta_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        elif self.method_embedding == 'BERT':
            BERT_model = transformers.TFBertModel.from_pretrained('bert-base-uncased')
            x = BERT_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        else:
            logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))

        # word vectors shape : (None, maxlen, 768)
        x = x[0]
        return x, inp
