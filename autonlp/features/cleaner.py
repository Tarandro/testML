import spacy

import pandas as pd

import re
import string

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

STOPWORDS = list(fr_stop)

from ..utils.logging import get_logger

logger = get_logger(__name__)


####################
# clean and spacy preprocessing
####################

def small_clean_text(text):
    """ Clean text : Remove '\n', '\r', URL, '’', numbers and double space
    Args:
        text (str)
    Return:
        text (str)
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('’', ' ', text)

    #text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def clean_text(text):
    """ Clean text : lower text + Remove '\n', '\r', URL, '’', numbers and double space + remove Punctuation
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
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    text = re.sub('’', ' ', text)

    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)

    return text


def nlp_preprocessing_spacy(data, nlp, disable_ner=False):
    """ nlp.pipe data preprocessing from a spacy model
    Args:
        data (pd.Series or List)
        nlp (Spacy model)
        disable_ner (Boolean) prediction of NER for each word
    Return:
        doc_spacy_data (List) each variable of the list is a Spacy object
    """
    list_content = list(data)
    if disable_ner:
        doc_spacy_data = [doc for doc in nlp.pipe(list_content, disable=["parser", "ner"])]
    else:
        doc_spacy_data = [doc for doc in nlp.pipe(list_content, disable=["parser"])]
    return doc_spacy_data


def transform_entities(doc_spacy_data):
    """ Named entities replacement (replace them with their general nouns) :
        Microsoft is replaced by <ORG>, or London by <LOC> / do not consider 'MISC' label
    Args:
        doc_spacy_data (List[spacy object]) documents preprocessing by spacy model
    Return:
        texts_with_entities (List[str]) documents with named entities replaced
    """
    texts_with_entities = []
    for doc in doc_spacy_data:
        if doc.ents == ():
            texts_with_entities.append(doc.text)
        else:
            doc_with_entities = doc.text
            for ent in doc.ents:
                if ent.label_ != 'MISC':
                    doc_with_entities = doc_with_entities.replace(ent.text, '<' + ent.label_ + '>')
            texts_with_entities.append(doc_with_entities)
    return texts_with_entities


def reduce_text_data(doc_spacy_data, keep_pos_tag, lemmatize):
    """ reduce documents with pos_tag and lemmatization + clean text at the end
    Args:
        doc_spacy_data (List[spacy object]): list of documents processed by nlp.pipe spacy
        keep_pos_tag (str or list): 'all' for no pos_tag else list of tags to keeps
        lemmatize (Boolean): apply lemmatization
    Return:
        data (List[str]) documents preprocessed
    """
    data = []
    for text in doc_spacy_data:
        if keep_pos_tag == 'all':
            if lemmatize:
                new_text = [token.lemma_ for token in text]
            else:
                new_text = [token.text for token in text]
        else:
            if lemmatize:
                new_text = [token.lemma_ for token in text if token.pos_ in keep_pos_tag]
            else:
                new_text = [token.text for token in text if token.pos_ in keep_pos_tag]
        data.append(clean_text(" ".join(new_text)))
    return data


#############################
#############################
#############################

class Preprocessing_NLP:
    """Class for compile full pipeline of NLP preprocessing task.
            Preprocessing_NLP steps:
                - create a Dataframe for targets (self.Y in __init__)
                - create a Dataframe with only the column with texts (self.data in __init__)
                - load spacy model according to name_spacy_model (self.nlp)
                - (Optional) can apply a small cleaning on text column
                - (Optional) preprocess text column with nlp.pipe spacy
                - (Optional) replace Named entities by tags
    """

    def __init__(self, data, flags_parameters):
        """
        Args:
            data (Dataframe)
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            target (str or list) : names of target columns
            apply_small_clean (Boolean) step 1 of transform
            apply_spacy_preprocessing (Boolean) step 2 of transform
            apply_entity_preprocessing (Boolean) step 3 of transform
        """
        self.data = data
        self.column_text = flags_parameters.column_text
        self.apply_small_clean = flags_parameters.apply_small_clean
        self.name_spacy_model = flags_parameters.name_spacy_model
        self.apply_spacy_preprocessing = flags_parameters.apply_spacy_preprocessing
        self.apply_entity_preprocessing = flags_parameters.apply_entity_preprocessing
        # you can't apply entity preprocessing if apply_spacy_preprocessing is False :
        if not self.apply_spacy_preprocessing:
            self.apply_entity_preprocessing = False
        self.last_spacy_model_download = None
        self.nlp = None
        self.map_label = dict()

        assert isinstance(self.data, pd.DataFrame), "data must be a DataFrame type"
        assert self.column_text in self.data.columns, 'column_text specifying the column with text is not in data'

        # self.target need to be a List
        self.target = flags_parameters.target
        if isinstance(self.target, list):
            self.target = self.target
        else:
            self.target = [self.target]

        if len([col for col in self.target if col in self.data.columns])> 0:
            self.Y = self.data[[col for col in self.target if col in self.data.columns]]
            assert self.Y.shape[1] > 0, 'target specifying the column with labels to predict is not in data'

            # create a map label if labels are not numerics
            self.map_label = flags_parameters.map_label
            if self.Y.shape[1] == 1 and self.map_label == {}:
                if not pd.api.types.is_numeric_dtype(self.Y.iloc[:, 0]):
                    self.map_label = {label: i for i, label in enumerate(self.Y.iloc[:, 0].unique())}
            if self.Y.shape[1] == 1 and self.map_label != {}:
                if self.Y[self.Y.columns[0]].iloc[0] in self.map_label.keys():
                    self.Y[self.Y.columns[0]] = self.Y[self.Y.columns[0]].map(self.map_label)
        else:
            self.Y = None

    def load_spacy_model(self, name_spacy_model="fr_core_news_md"):
        """ Download Spacy pre-train model
        Args:
            name_spacy_model (str)
        """
        if self.apply_spacy_preprocessing:
            if '/' not in name_spacy_model:
                spacy.cli.download(name_spacy_model)
            if name_spacy_model != self.last_spacy_model_download:
                try:
                    self.nlp = spacy.load(name_spacy_model)
                    self.last_spacy_model_download = name_spacy_model
                except Exception:
                    logger.error("unknown spacy model name")

    def transform(self, data):
        """ Fit and transform self.data :
            + can apply a small cleaning on text column (self.apply_small_clean)
            + preprocess text column with nlp.pipe spacy (self.apply_spacy_preprocessing)
            + replace Named entities  (self.apply_entity_preprocessing)
        Return:
            self.data (DataFrame) only have one column : column_text
        """

        data_copy = data.copy()

        self.load_spacy_model(self.name_spacy_model)

        if self.apply_small_clean:
            logger.info("- Apply small clean of texts...")
            data_copy[self.column_text] = data_copy[self.column_text].apply(lambda text: small_clean_text(text))

        if self.apply_spacy_preprocessing:
            logger.info("- Apply nlp.pipe from spacy...")
            doc_spacy_data = nlp_preprocessing_spacy(data_copy[self.column_text], self.nlp,
                                                          disable_ner=self.apply_entity_preprocessing)
        else:
            doc_spacy_data = None

        if self.apply_entity_preprocessing:
            logger.info("- Apply entities preprocessing...")
            data_copy[self.column_text] = transform_entities(doc_spacy_data)
            doc_spacy_data = nlp_preprocessing_spacy(data_copy[self.column_text], self.nlp, disable_ner=True)

        # keep only the column with texts
        return data_copy[[self.column_text]], doc_spacy_data
