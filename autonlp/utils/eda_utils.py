import pandas as pd
import re
import string
import random as rd

from dataclasses import dataclass, field
from typing import Dict


from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt

"""Logging."""
import logging
from ..utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


@dataclass
class Flags_EDA:
    """ Class to instantiate parameters """
    ### General
    # path for csv data, use for train/test split:
    path_data: str = field(default_factory=str)
    # use manual logs:
    apply_logs: bool = True
    # outdir of the manual logs:
    outdir: str = "./logs"
    # for debug : use only 50 data rows for training
    debug: bool = False
    # verbosity levels:`0`: No messages / `1`: Warnings / `2`: Info / `3`: Debug.
    verbose: int = 2

    # name of the column with text
    column_text: str = 'text'
    # language of the text, 'fr' or 'en'
    language_text: str = 'fr'
    # name of a column that groups data
    column_group: str = field(default_factory=str)
    # name of a column with date
    column_date: str = field(default_factory=str)
    # name of a column with sentiment labels
    column_sentiment: str = field(default_factory=str)
    # name of a column with confidence about sentiments
    column_confidence: str = field(default_factory=str)

    ### Preprocessing
    # can apply a small cleaning on text column:
    apply_small_clean: bool = True
    # name of spacy model for preprocessing (fr:"fr_core_news_md", en:"en_core_web_md")
    name_spacy_model: str = "fr_core_news_md"

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


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

    # text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def clean_text(text):
    """ Clean text : lower text + Remove '\n', '\r', URL, '’', emoji, numbers and double space + remove Punctuation
    Args:
        text (str)
    Return:
        text (str)
    """
    text = str(text).lower()

    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('\xa0', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    text = re.sub('’', ' ', text)

    # text = re.sub(' \d+', ' ', text)
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
                                        u"\ufe0f"  # dingbats
                                        u"\u3030"  # flags (iOS)
                                        "]+", flags=re.UNICODE)

    text = regrex_pattern.sub(r'', text)

    return text


def remove_stopwords(text, STOPWORDS):
    return (" ".join([w for w in text.lower().split() if w not in STOPWORDS if len(w) > 1]))


class Preprocessing_EDA:

    def __init__(self, data, flags_parameters):
        """
        Args:
            data (Dataframe)
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            apply_small_clean (Boolean) step 1 of transform
            apply_spacy_preprocessing (Boolean) step 2 of transform
            language_text (str) language 'fr' or 'en'
        """
        self.data = data
        self.column_text = flags_parameters.column_text
        self.apply_small_clean = flags_parameters.apply_small_clean
        self.language_text = flags_parameters.language_text

        assert isinstance(self.data, pd.DataFrame), "data must be a DataFrame type"
        assert self.column_text in self.data.columns, 'column_text specifying the column with text is not in data'

    def fit_transform(self):
        """ Fit and transform self.data :
            + can apply a small cleaning on text column (self.apply_small_clean)
            + create a column text : 'clean_text' from clean_text function
            + create a column clean_rsw_text : 'clean_text' + remove stop words
        Return:
            self.data (DataFrame) only have one column : column_text
        """

        if self.apply_small_clean:
            logger.info("- Apply small clean of texts...")
            self.data[self.column_text] = self.data[self.column_text].apply(lambda text: small_clean_text(text))

        logger.info("- Create a column clean_text: apply_clean_text...")
        self.data['clean_text'] = self.data[self.column_text].apply(lambda text: clean_text(text))

        if self.language_text == 'fr':
            from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
            STOPWORDS = list(fr_stop) + ['qu']
        else:
            from spacy.lang.en.stop_words import STOP_WORDS as en_stop
            STOPWORDS = list(en_stop)

        logger.info("- Create a column clean_rsw_text: apply_clean_text+remove_stopwords...")
        self.data['clean_rsw_text'] = self.data['clean_text'].apply(lambda text: remove_stopwords(text, STOPWORDS))
        return self.data