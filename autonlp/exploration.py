import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
from collections import defaultdict
from raceplotly.plots import barplot
import datetime
from plotly.subplots import make_subplots
import re
import string
import random as rd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import plotly.graph_objs as go
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

from .utils.eda_utils import *

"""Logging."""
import logging
from .utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Eda_NLP:
    """Class for compile full pipeline of EDA task on text data"""

    def __init__(self, flags_parameters):
        """
        Args:
            flags_parameters : Instance of Flags class object

        From flags_parameters:
            apply_logs (Boolean) : use manual logs
            outdir (str) : path of output logs
            column_text (str) : name of the column with texts (only one column)
            column_date (str) : name of the column with date
            column_sentiment (str) : name of the column with sentiment labels
            column_confidence (str) : name of the column with sentiment scores
            verbose (int)
        """
        self.flags_parameters = flags_parameters

        self.apply_logs = flags_parameters.apply_logs
        self.outdir = flags_parameters.outdir
        self.column_text = flags_parameters.column_text
        self.column_date = flags_parameters.column_date
        self.column_sentiment = flags_parameters.column_sentiment
        self.column_confidence = flags_parameters.column_confidence

        self.verbose = flags_parameters.verbose
        logging.getLogger().setLevel(verbosity_to_loglevel(self.verbose))

    def data_preprocessing(self, dataset=None, copy=True):
        """ Apply :class:Preprocessing_NLP
        Args :
            data (Dataframe)
        """
        if copy:
            data = dataset.copy()
        else:
            data = dataset
        # Read data
        if data is None:
            logger.info("\nRead data...")
            data = pd.read_csv(self.flags_parameters.path_data)

        data[self.column_text] = data[self.column_text].fillna('')

        if self.flags_parameters.debug:
            data = data.iloc[:50]

        # Preprocessing
        logger.info("\nBegin preprocessing of {} data :".format(len(data[self.column_text])))
        self.pre = Preprocessing_EDA(data, self.flags_parameters)
        self.data = self.pre.fit_transform()

    def show_number_docs(self):
        """ Number of documents/rows in column 'flags_parameters.column_text' """
        number_texts = len(self.data[~self.data[self.column_text].isin([''])][self.column_text])
        logger.info("Number of non empty texts in columns '{}' : {}".format(self.column_text, number_texts))

    def show_number_unique_words(self, column_text="clean_text"):
        """ Number of unique words in column 'column_text' """
        text = " ".join(self.data[column_text])
        text = re.sub(' \d+', ' ', text)
        number_unique_words = len(set(text.split(' ')))
        logger.info("Number of unique words in columns '{}' : {}".format(column_text, number_unique_words))

    def show_range_date(self):
        """ Range minimum date to maximum date """
        if self.column_date in list(self.data.columns):
            logger.info('Documents dating between {} and {}'.format(self.data[self.column_date].min().date(),
                                                                    self.data[self.column_date].max().date()))

    def show_top_n_words(self, column_text="clean_rsw_text", n=10):
        """ Top n 1-grams in column 'column_text' """
        df_ngrams = self.count_ngrams(column_text, 1)[:n]
        logger.info("Top {} words : {}".format(n, list(df_ngrams.terms)))

    def show_summary(self):
        """ Summary: apply 4 last functions """
        self.show_number_docs()
        self.show_number_unique_words()
        self.show_range_date()
        self.show_top_n_words()

    def show_random_example(self, columns='all'):
        """ For a random row of data : show each column in 'columns' """
        if columns == "all":
            columns = list(self.data.columns)
        if isinstance(columns, str):
            columns = list(columns)

        idx = rd.choice([i for i in range(len(self.data))])

        subset_data = self.data.iloc[idx]

        logger.info("row {}".format(idx))

        for col in columns:
            if col in self.data.columns:
                logger.info("\n{} : {}".format(col, subset_data[col]))

    def show_number_docs_bygroup(self, column_group=None):
        """ Plot a pie for the column 'column_group' (max 10 values) """
        if column_group is None:
            column_group = self.flags_parameters.column_group
        if column_group not in self.data.columns:
            return
        return px.pie(pd.DataFrame(self.data[column_group].value_counts()).reset_index()[:10],
                      values=column_group, names='index',
                      title='Number of documents by {}'.format(column_group), template='plotly_dark')

    def show_number_of_words_bydocs(self, column_text="clean_text"):
        """ Plot histogram : Average word count per document in column 'column_text' """
        return px.histogram(self.data[column_text].apply(lambda x: len(x.split(' '))),
                            template='plotly_dark', title="Average word count per document")

    def count_ngrams(self, column_text="clean_rsw_text", n_gram=2):
        """ Return a dataframe with all n_gram in column 'column_text' with a count """
        ngrams = defaultdict(int)

        def generate_ngrams(text, n_gram=1):
            token = [token for token in
                     text.lower().split(' ')]  # if token != '' if token not in STOPWORDS if len(token) > 2]
            ngrams = zip(*[token[i:] for i in range(n_gram)])
            return [' '.join(ngram) for ngram in ngrams]

        for text in self.data[column_text]:
            for word in generate_ngrams(text, n_gram):
                ngrams[word] += 1

        df_ngrams = pd.DataFrame(sorted(ngrams.items(), key=lambda x: x[1])[::-1], columns=['terms', 'count'])

        return df_ngrams

    def show_ngrams_frequency(self, column_text="clean_rsw_text", n_gram=2, limit=20):
        """ Frequency barplot of Top 'limit' n_gram in column 'column_text'  """
        df_ngrams = self.count_ngrams(column_text, n_gram)[:limit]
        return px.bar(df_ngrams, x='terms', y='count', template='plotly_dark',
                      title="Frequency of Top {}-grams".format(n_gram))

    def show_wordcloud(self, apply_tfidf=False, column_text="clean_rsw_text", collocation_threshold=10, width=1500,
                       height=800, figsize_x=18, figsize_y=8):
        """ Worcloud with words in column 'column_text', decrease 'collocation_threshold' to get more bigrams """

        if apply_tfidf:
            corpus = list(self.data[column_text])
            vectorizer = TfidfVectorizer()
            vecs = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names()
            dense = vecs.todense()
            lst1 = dense.tolist()
            df = pd.DataFrame(lst1, columns=feature_names)
            wordcloud = WordCloud(width=width, height=height,
                                  collocation_threshold=collocation_threshold).generate_from_frequencies(
                df.T.sum(axis=1))
        else:
            text = " ".join(self.data[column_text])
            wordcloud = WordCloud(width=width, height=height, collocation_threshold=collocation_threshold).generate(
                text)
        plt.figure(figsize=(figsize_x, figsize_y))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def show_information_terms(self, terms=None, column_text="clean_text"):
        """ Show frequency of specific terms in column 'column_text' """
        if terms is None:
            return
        elif isinstance(terms, str):
            terms = [terms]

        for term in terms:
            n_documents = 0
            n_count = 0
            for doc in self.data[column_text]:
                count = doc.count(term)
                if count != 0:
                    n_count += count
                    n_documents += 1
            logger.info("Term '{}' appears in {} documents with a total count of {}".format(term, n_documents, n_count))

    def show_explained_variance(self, column_text="clean_rsw_text", n_components=500, n_top_words=10):
        """ Information about explained variance with SVD reduction dimension in 'n_components' dimension for column 'column_text' """
        CVZ = CountVectorizer()
        SVD = TruncatedSVD(n_components)

        C_vector = CVZ.fit_transform(self.data[column_text])
        pc_matrix = SVD.fit_transform(C_vector)

        evr = SVD.explained_variance_ratio_
        total_var = evr.sum() * 100
        cumsum_evr = np.cumsum(evr)

        trace1 = {"name": "individual explained variance", "type": "bar", 'y': evr}
        trace2 = {"name": "cumulative explained variance", "type": "scatter", 'y': cumsum_evr}
        data_trace = [trace1, trace2]
        layout = {"xaxis": {"title": "Principal components"}, "yaxis": {"title": "Explained variance ratio"}}
        fig = go.Figure(data=data_trace, layout=layout)
        fig.update_layout(
            title='{:.2f}% of the text variance of documents can be explained with {} words'.format(total_var,
                                                                                                    len(evr)))
        fig.show()

        best_features = [[CVZ.get_feature_names()[i], SVD.components_[0][i]] for i in
                         SVD.components_[0].argsort()[::-1][:n_top_words]]
        worddf = pd.DataFrame(np.array(best_features[:n_top_words])[:, 0]).rename(columns={0: 'Word'})
        worddf['Explained Variance'] = np.round(evr[:n_top_words] * 100, 2)
        worddf['Explained Variance'] = worddf['Explained Variance'].apply(lambda x: str(x) + '%')
        app = []
        for word in worddf.Word:
            total_count = 0
            for doc in self.data[column_text]:
                if doc.find(word) != -1:
                    total_count += 1
            app.append(total_count)
        worddf['Appeared_On_X_docs'] = app

        fig = go.Figure()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Word<b>', "<b>Explains X% of Variance<b>",
                            '<b>Appears in X documents ({} documents)<b>'.format(len(self.data[column_text]))],
                    font=dict(size=19, family="Lato"),
                    align="center"
                ),
                cells=dict(
                    values=[worddf[k].tolist() for k in ['Word', "Explained Variance", 'Appeared_On_X_docs']],
                    align="center")
            ))
        fig.show()

    def show_number_docs_by_date(self, n_rolling_days=0):
        """ plot Line graph : Number of documents over time (rolling day option) """
        if self.column_date not in self.data.columns:
            return
        df = self.data.copy()
        df = df[~df[self.column_text].isin([''])]
        df['day'] = self.data[self.column_date].apply(lambda x: x.date())
        grouped_data = df.groupby(['day']).count()
        df = pd.DataFrame()
        df['day'] = np.array(list(grouped_data.index))
        df['count'] = grouped_data.iloc[:, 0].values

        if n_rolling_days > 0:
            df['count'] = df['count'].rolling(n_rolling_days).mean()
            title = "Number of documents over time (rolling {} days)".format(n_rolling_days)
        else:
            title = 'Number of documents over time'

        fig = px.line(df, x="day", y="count",
                      title=title, labels={'day': 'date'})
        fig.show()

    def show_average_doc_by_day(self):
        """ plot bar graph : Average number of documents per day of the week """
        if self.column_date not in self.data.columns:
            return
        df = self.data.copy()
        df = df[~df[self.column_text].isin([''])]
        df['day'] = self.data[self.column_date].apply(lambda x: x.date())
        grouped_data = df.groupby(['day']).count()
        df = pd.DataFrame()
        df['day'] = np.array(list(grouped_data.index))
        df['count'] = grouped_data.iloc[:, 0].values

        dict_day = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['weekday'] = df['day'].apply(lambda x: x.weekday()).map(dict_day)

        df = df.groupby(['weekday']).mean().reset_index()

        fig = px.bar(df, x='weekday', y='count', color='weekday', template='plotly_dark',
                     category_orders={
                         "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                     title="Average number of documents per day of the week")
        fig.layout.update(showlegend=False)
        fig.show()

    ####################
    # Sentiment Analysis
    ####################

    def apply_textblob_sentiment_analysis(self):
        """ French sentiment Analysis with TextBlob : use sentiment score of each word to get sentiment and confidence of a document : Négatif/Neutre/Positif """
        tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

        from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
        STOPWORDS = list(fr_stop) + ['qu']
        STOPWORDS_textblob = [sw for sw in STOPWORDS if
                              sw not in ['n\'', 'ne', 'pas', 'plus', 'personne', 'aucun', 'ni', 'aucune', 'rien']]

        def remove_stop_words(text):
            new_text = [w for w in text.split() if not ((w in STOPWORDS_textblob) or (len(w) == 1))]
            return ' '.join(new_text)

        def analyse_sentiment_textblob(text):
            score_sentiment = tb(text).sentiment[0]
            return score_sentiment

        def label_sentiment(score_sentiment):
            if (score_sentiment > 0):
                return 'Positif'
            elif (score_sentiment < 0):
                return 'Négatif'
            else:
                return 'Neutre'

        self.data['confidence'] = self.data['clean_text'].apply(lambda text: remove_stop_words(text)).apply(
            lambda text: analyse_sentiment_textblob(text))
        self.data['sentiment'] = self.data.confidence.apply(lambda x: label_sentiment(x))
        count_sentiments = self.data['sentiment'].value_counts()
        if 'Positif' in count_sentiments: logger.info(
            "Number of Positive sentiment documents : {}".format(count_sentiments['Positif']))
        if 'Neutre' in count_sentiments: logger.info(
            "Number of Neutral sentiment documents  : {}".format(count_sentiments['Neutre']))
        if 'Négatif' in count_sentiments: logger.info(
            "Number of Negative sentiment documents : {}".format(count_sentiments['Négatif']))

    def show_histogram_sentiment(self, column_sentiment="sentiment"):
        """ Plot histogram of number of documents per sentiment """
        if self.column_sentiment is not None:
            column_sentiment = self.column_sentiment
        if column_sentiment not in self.data.columns:
            self.apply_textblob_sentiment_analysis()
            column_sentiment = "sentiment"

        color_discrete_map = {}
        if "Négatif" in self.data[column_sentiment].unique():
            color_discrete_map["Négatif"] = "red"
        if "Neutre" in self.data[column_sentiment].unique():
            color_discrete_map["Neutre"] = "grey"
        if "Positif" in self.data[column_sentiment].unique():
            color_discrete_map["Positif"] = "green"

        fig = px.histogram(self.data, x=column_sentiment, color=column_sentiment, template='plotly_dark',
                           title="Number of documents per sentiment",
                           color_discrete_map=color_discrete_map)
        fig.layout.update(showlegend=False)
        fig.show()

    def show_wordcloud_by_sentiment(self, apply_tfidf=False, pourcentage=0, rate_color=1.5,
                                    column_sentiment="sentiment", column_confidence="confidence",
                                    column_text="clean_rsw_text",
                                    word_to_remove=[]):
        """ Show a wordcloud for texts classify as 'Positif' and another wordcloud for 'Négatif' texts """
        if self.column_confidence is not None:
            column_confidence = self.column_confidence
        if self.column_sentiment is not None:
            column_sentiment = self.column_sentiment
        if column_sentiment not in self.data.columns:
            self.apply_textblob_sentiment_analysis()
            column_sentiment = "sentiment"
            column_confidence = "confidence"

        df_total = pd.DataFrame()
        titles = {}

        if "Négatif" in self.data[column_sentiment].unique() and "Positif" in self.data[column_sentiment].unique():
            fig, ax = plt.subplots(2, 1, figsize=(16, 8))

            df_total = pd.DataFrame()

            for i, sent in enumerate(['Négatif', 'Positif']):
                dd = self.data[self.data[column_sentiment] == sent]
                if column_confidence in self.data.columns and pourcentage > 0:
                    limit = np.quantile(dd[column_confidence], 1 - pourcentage)
                    dd = dd[np.abs(dd[column_confidence]) > np.abs(limit)]
                    titles[sent] = "Wordcloud : Top {}% most '{}' documents ".format(pourcentage * 100, sent)
                else:
                    titles[sent] = sent

                if apply_tfidf:
                    vectorizer = TfidfVectorizer()
                else:
                    vectorizer = CountVectorizer()
                corpus = list(dd[column_text])
                vecs = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names()
                dense = vecs.todense()
                lst1 = dense.tolist()
                df = pd.DataFrame(lst1, columns=feature_names)
                df = df.T.sum(axis=1)
                if len(word_to_remove) > 0:
                    df = df[~df.index.isin(word_to_remove)]
                df = pd.DataFrame(df, columns=[sent])
                df_total = pd.concat([df_total, df], axis=1).fillna(0)

            default_color = 'grey'
            color_to_words = {'#00ff00': [], "red": []}
            for i, row in df_total.iterrows():
                if row['Négatif'] > row['Positif'] * rate_color:
                    color_to_words['red'].append(row.name)
                elif row['Positif'] > row['Négatif'] * rate_color:
                    color_to_words['#00ff00'].append(row.name)

            for i, sent in enumerate(['Négatif', 'Positif']):
                wordcloud = WordCloud(width=1500, height=500).generate_from_frequencies(df_total[sent])

                grouped_color_func = GroupedColorFunc(color_to_words, default_color)
                wordcloud.recolor(color_func=grouped_color_func)

                ax[i].imshow(wordcloud, interpolation='bilinear')
                ax[i].set_title(titles[sent])
                ax[i].set_axis_off()
            plt.savefig('Sentiment_wordcloud.png')

    def show_sentiment_score_by_day(self, column_confidence="confidence", n_rolling_days=0):
        """ Plot line graph: Average sentiment score per day over time (rolling day option) """
        if self.column_date not in self.data.columns:
            return
        if self.column_confidence is not None:
            column_confidence = self.column_confidence
        if column_confidence not in self.data.columns:
            self.apply_textblob_sentiment_analysis()
            column_confidence = "confidence"
        df = self.data.copy()
        df = df[~df[self.column_text].isin([''])]
        df['day'] = self.data[self.column_date].apply(lambda x: x.date())
        df = df[['day', column_confidence]].groupby(['day']).mean().reset_index()

        if n_rolling_days > 0:
            df['rolling_confidence'] = df[column_confidence].rolling(n_rolling_days).mean()
            column_confidence = "rolling_confidence"
            title = "Average sentiment score (rolling {} days)".format(n_rolling_days)
        else:
            title = 'Average sentiment score per day'

        fig = px.line(df, x="day", y=column_confidence,
                      title=title, labels={'day': 'date', column_confidence: "confidence"})
        fig.add_shape(type="line",
                      x0=df['day'].values[0], y0=df[column_confidence].mean(), x1=df['day'].values[-1],
                      y1=df[column_confidence].mean(),
                      line=dict(color="Red", width=2, dash="dashdot"), name='Mean')
        fig.show()

    def show_number_sentiment_by_day(self, column_sentiment="sentiment", n_rolling_days=0):
        """ Plot line graph: Number of documents per day group by sentiment over time (rolling day option) """
        if self.column_date not in self.data.columns:
            return
        if self.column_sentiment is not None:
            column_sentiment = self.column_sentiment
        if column_sentiment not in self.data.columns:
            self.apply_textblob_sentiment_analysis()
            column_sentiment = "sentiment"
        df = self.data.copy()
        df = df[~df[self.column_text].isin([''])]
        df['day'] = self.data[self.column_date].apply(lambda x: x.date())
        grouped_data = df.groupby(['day', column_sentiment]).count()

        ind = np.array(list(grouped_data.index))
        df = pd.DataFrame()
        df['day'] = ind[:, 0]
        df[column_sentiment] = ind[:, 1]
        df['count'] = grouped_data.iloc[:, 0].values

        if n_rolling_days > 0:
            df = df.pivot(index='day', columns=column_sentiment, values='count').fillna(0).rolling(
                n_rolling_days).mean().reset_index()
            title = "Number of documents group by sentiment over time (rolling {} days)".format(n_rolling_days)
            y = df.columns
        else:
            title = 'Number of documents group by sentiment over time'
            y = "count"

        color_discrete_map = {}
        if "Négatif" in self.data[column_sentiment].unique():
            color_discrete_map["Négatif"] = "red"
        if "Neutre" in self.data[column_sentiment].unique():
            color_discrete_map["Neutre"] = "grey"
        if "Positif" in self.data[column_sentiment].unique():
            color_discrete_map["Positif"] = "green"

        fig = px.line(df, x="day", y=y, color=column_sentiment,
                      title=title, labels={'day': 'date'},
                      color_discrete_map=color_discrete_map)
        fig.update_xaxes()
        fig.show()

    def show_readme(self):
        readme = "Possible EDA methods :\n"
        readme += "\n---Summary EDA:\n"
        readme += "-show_number_docs : Number of documents/rows in column 'flags_parameters.column_text'\n"
        readme += "-show_number_unique_words : Number of unique words in column 'column_text'\n"
        readme += "-show_range_date : Range minimum date to maximum date\n"
        readme += "-show_top_n_words : Top n 1-grams in column 'column_text'\n"
        readme += "-show_summary : Summary: apply 4 last functions\n"

        readme += "\n---Frequency EDA:\n"
        readme += "-show_random_example : For a random row of data : show each column in 'columns'\n"
        readme += "-show_number_docs_bygroup : Plot a pie for the column 'column_group' (max 10 values)\n"
        readme += "-show_number_of_words_bydocs : Plot histogram : Average word count per document in column 'column_text'\n"
        readme += "-count_ngrams : Return a dataframe with all n_gram in column 'column_text' with a count\n"
        readme += "-show_ngrams_frequency : Frequency barplot of Top 'limit' n_gram in column 'column_text'\n"
        readme += "-show_wordcloud : Worcloud with words in column 'column_text', decrease 'collocation_threshold' to get more bigrams\n"
        readme += "-show_information_terms : Show frequency of specific terms in column 'column_text'\n"
        readme += "-show_explained_variance : Information about explained variance with SVD reduction dimension in 'n_components' dimension for column 'column_text'\n"

        readme += "\n---Date Frequency EDA:\n"
        readme += "-show_number_docs_by_date : plot Line graph : Number of documents over time (rolling day option)\n"
        readme += "-show_average_doc_by_day : plot bar graph : Average number of documents per day of the week\n"

        readme += "\n---Sentiment Analysis EDA:\n"
        readme += "-apply_textblob_sentiment_analysis : French sentiment Analysis with TextBlob : use sentiment score of each word to get sentiment and confidence of a document : Négatif/Neutre/Positif\n"
        readme += "-show_histogram_sentiment : Plot histogram of number of documents per sentiment\n"
        readme += "-show_wordcloud_by_sentiment : Show a wordcloud for texts classify as 'Positif' and another wordcloud for 'Négatif' texts\n"

        readme += "\n---Date Sentiment Analysis EDA:\n"
        readme += "-show_sentiment_score_by_day : Plot line graph: Average sentiment score per day over time (rolling day option)\n"
        readme += "-show_number_sentiment_by_day : Plot line graph: Number of documents per day group by sentiment over time (rolling day option)\n"

        logger.info(readme)