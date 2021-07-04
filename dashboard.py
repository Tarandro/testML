import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from autonlp.exploration_streamlit import Eda_NLP
from autonlp.utils.eda_utils import Flags_EDA

### run application : streamlit run dashboard.py

################
# Load data
################


st.header('AutoNLP')

url = 'http://127.0.0.1:5000/predict'

Section = st.sidebar.radio(
    'Section :', ['Exploration', 'Prediction'])


if Section == 'Prediction':
    params = dict()

    # displays a file uploader widget
    data = st.file_uploader("Choose data", type=['csv'])

    user_input = st.text_input("Text :", "Le marché Français a diminué de 50%")

    btn = st.button('Prediction')
    if btn:
        info_pred = st.empty()
        info_pred.text('Running ...')
        if data is None:
            params['text'] = user_input
            resp = requests.post(url, files=params)
            dict_prediction = resp.json()
        else:
            params['csv_file'] = data.getbuffer()
            resp = requests.post(url, files=params)
            dict_prediction = resp.json()
            print(dict_prediction)

        info_pred.text('Prediction : ')
        st.json(dict_prediction)


else:
    data_file = st.file_uploader("Choose data to explore", type=['csv'])


    if data_file is not None:

        info_pred = st.empty()
        info_pred.text('Reading...')

        data = pd.read_csv(data_file)
        info_pred.text('')

        col_name = list(data.columns)

        index = 0
        for i, col in enumerate(col_name):
            if "text" in col.lower():
                index = i+1
        column_text = st.selectbox("column_text :", [None] + col_name, index=index)
        language_text = st.selectbox("language_text :", ["fr", "en"])
        index = 0
        for i, col in enumerate(col_name):
            if "group" in col.lower():
                index = i+1
        column_group = st.selectbox("column_group :", [None] + col_name, index=index)
        index = 0
        for i, col in enumerate(col_name):
            if "date" in col.lower():
                index = i + 1
        column_date = st.selectbox("column_date (format : %Y-%m-%d):", [None] + col_name, index=index)
        index = 0
        for i, col in enumerate(col_name):
            if "sentiment" in col.lower():
                index = i + 1
        column_sentiment = st.selectbox("column_sentiment :", [None] + col_name, index=index)
        index = 0
        for i, col in enumerate(col_name):
            if "confidence" in col.lower():
                index = i + 1
        column_confidence = st.selectbox("column_confidence :", [None] + col_name, index=index)

        btn = st.button('Exploration')

        if btn:
            flags_dict_info = {"column_text": column_text, "language_text": language_text, "column_group": column_group,
                               "column_date": column_date, "column_sentiment": column_sentiment,
                               "column_confidence": column_confidence}
            flags = Flags_EDA().update(flags_dict_info)

            eda = Eda_NLP(flags)
            eda.data_preprocessing(data)

            summary = eda.show_summary()
            st.text(summary)

            st.write(eda.show_number_docs_bygroup())

            st.write(eda.show_number_of_words_bydocs())

            st.write(eda.show_ngrams_frequency())

            wc = eda.show_wordcloud(apply_tfidf=True)
            if wc is not None:
                st.write("Wordcloud")
            st.write(wc)

            fig, fig2 = eda.show_explained_variance()
            if fig is not None:
                st.write("Explained Variance")
            st.write(fig)
            if fig is not None:
                st.write("Explained Variance Top words")
            st.write(fig2)

            st.write(eda.show_number_docs_by_date(n_rolling_days=30))

            st.write(eda.show_average_doc_by_day())

            st.write(eda.show_histogram_sentiment())

            wc = eda.show_wordcloud_by_sentiment(apply_tfidf=True)
            if wc is not None:
                st.write("Wordcloud by sentiment")
            st.write(wc)

            st.write(eda.show_sentiment_score_by_day())

            st.write(eda.show_sentiment_score_by_day(n_rolling_days=15))

            st.write(eda.show_number_sentiment_by_day())

            st.write(eda.show_number_sentiment_by_day(n_rolling_days=15))

