from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags


#####################
# Parameters
#####################

flags_dict_info = {
    "debug": False,  # for debug : use only 50 data rows for training
    "path_data": "C:/Users/agassmann/Documents/data/FinancialPhraseBank.csv",
    "path_data_validation": "",
    "apply_logs": True,
    "outdir": "./logs",
    "apply_mlflow": False,
    "experiment_name": "AutoNLP_3",
    "seed": 15,
    "apply_app": False
}
flags_dict_preprocessing = {
    "column_text": "text_fr",  # name column with texts
    "target": "sentiment",
    "language_text": "fr",
    "apply_small_clean": True,
    "name_spacy_model": "fr_core_news_md",  # en_core_web_md
    "apply_spacy_preprocessing": False,
    "apply_entity_preprocessing": True
}

flags_dict_autonlp = {
    "objective": 'multi-class',    # 'binary' or 'multi-class' or 'regression'
    "include_model": ['doc2vec'],  # 'tf', 'tf-idf', 'doc2vec'
    "max_run_time_per_model": 10,
    "frac_trainset": 0.7,
    "scoring": 'f1',
    "nfolds": 5,
    "nfolds_train": 1,
    "class_weight": True,
    "apply_blend_model": True,
    "verbose": 2,
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'CamemBERT',
                         'spacy': [('all', False)]},

    "apply_optimization": True,
    "apply_validation": True,
    "path_models_parameters": None,
    "path_models_best_parameters": None
}

flags_dict_display = {
    "sort_leaderboard": 'f1'
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
print("flags :", flags)
debug = flags.debug

# mlflow : (mlflow ui --backend-store-uri ./mlruns)

if __name__ == '__main__':
    #####################
    # AutoNLP
    #####################

    autonlp = AutoNLP(flags)

    #####################
    # Preprocessing
    #####################

    autonlp.data_preprocessing()

    #####################
    # Training
    #####################

    dict_embeddings_train = autonlp.fit_transform_embedding()

    #####################
    # Testing
    #####################

    on_test_data = True
    name_logs = 'last_logs'
    dict_embeddings_test = autonlp.transform_embedding(name_logs=name_logs, on_test_data=on_test_data)
