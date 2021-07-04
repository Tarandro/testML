import pandas as pd

import re
import string


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