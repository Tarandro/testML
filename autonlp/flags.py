from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Any, Union
from yaml import dump, full_load


@dataclass
class Flags:
    """ Class to instantiate parameters """
    ### General
    # path for csv data, use for train/test split:
    path_data: str = field(default_factory=str)
    # (Optional) path csv data to use for validation instead of cross-validation:
    path_data_validation: str = field(default_factory=str)
    # use manual logs:
    apply_logs: bool = True
    # outdir of the manual logs:
    outdir: str = "./logs"
    # use MLflow Tracking (save in "./mlruns")
    # dashboard : ($ mlflow ui --backend-store-uri ./mlruns)
    apply_mlflow: bool = False
    # MLflow Experiment name
    experiment_name: str = "Experiment"
    # if you want to use on model from model_deployment directory
    apply_app: bool = False
    # for debug : use only 50 data rows for training
    debug: bool = False
    # seed use for train/test split, cross-validation split and folds choice:
    seed: int = 15

    # name of the column with text
    column_text: str = 'text'
    # language of the text, 'fr' or 'en'
    language_text: str = 'fr'
    # name of target column
    target: str = 'target'

    ### Preprocessing ML
    ordinal_features: list = field(default_factory=list)
    normalize: bool = True
    method_scaling: str = 'MinMaxScaler'   # 'MinMaxScaler', 'RobustScaler', 'StandardScaler'
    type_columns: dict = field(default_factory=lambda: None)
    apply_preprocessing_mandatory: bool = True
    remove_categorical: bool = False

    method_nan_categorical: str = 'constant'
    method_nan_numeric: str = 'mean'
    subsample: float = 1
    feature_interaction: bool = False
    feature_ratio: bool = False
    polynomial_features: bool = False
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    feature_selection: bool = False
    feature_selection_threshold: float = 0.8
    bin_numeric_features: list = field(default_factory=list)
    remove_low_variance: bool = False
    remove_percentage: float = 0.8
    info_pca: dict = field(default_factory=dict)
    info_tsne: dict = field(default_factory=dict)
    info_stats: dict = field(default_factory=dict)

    ### Preprocessing Time-Series
    startDate_train: str = 'all'  # or int  need to be a continuous numeric column
    endDate_train: str = 'all'    # or int
    position_id: str = field(default_factory=lambda: None)   # can be a dataframe
    position_date: str = field(default_factory=lambda: None)   # need to be a continuous numeric column
    size_train_prc: float = 0.8
    time_series_recursive: bool = False
    LSTM_date_features: list = field(default_factory=list)
    step_lags: list = field(default_factory=list)
    step_rolling: list = field(default_factory=list)
    win_type: str = field(default_factory=lambda: None)

    ### Preprocessing NLP
    # can apply a small cleaning on text column:
    apply_small_clean: bool = True
    # name of spacy model for preprocessing (fr:"fr_core_news_md", en:"en_core_web_md")
    name_spacy_model: str = "fr_core_news_md"
    # preprocess text column with nlp.pipe spacy:
    apply_spacy_preprocessing: bool = False
    # replace Named entities by tags:
    apply_entity_preprocessing: bool = False

    ### AutoNLP
    # specify target objective : 'binary' / 'multi-class' / 'regression'
    objective: str = 'binary'
    # list of name models to include :
    include_model: list = field(default_factory=list)
    # maximum Hyperparameters Optimization time in seconds for each model:
    max_run_time_per_model: int = 60
    # train/test split fraction:
    frac_trainset: float = 0.7
    # number of folds to split during optimization and cross-validation :
    nfolds: int = 5
    # number of folds to train during optimization and cross-validation :
    nfolds_train: int = 5
    # metric to optimize during Hyperparameters Optimization :
    # binary : ['accuracy','f1','recall','precision','roc_auc']
    # multi-class : ['accuracy','f1','recall','precision']
    # regression : ['mse','explained_variance','r2']
    scoring: str = "accuracy"
    # Only for multi-class : method to average scoring : ['micro', 'macro', 'weighted']
    average_scoring: str = "weighted"
    # cross-validation method : 'StratifiedKFold' or 'KFold'
    cv_strategy: str = "KFold"
    # apply a weight for each class during the compute of losses
    class_weight: bool = False
    # size of parameters range for optimization : 'small' or big' (not implemented)
    size_params: str = 'small'
    # information about embedding method:
    # For 'Word2Vec', 'FastText' and 'Doc2Vec', you can create a gensim model from scratch by writing
    #     the name of the model as embedding method or you can use pre-trained model/weights by indicating the path.
    # For 'Transformer', you have the choice between these pre-trained models : 'BERT', 'RoBERTa',
    #     'CamemBERT', 'FlauBERT' and 'XLM-RoBERTa'
    # For 'Spacy', it doesn't indicate an embedding method but the preprocessing step for
    #     tf and tf-idf embedding method. You can choose several preprocessing methods in the tuple
    #     format (keep_pos_tag, lemmatize). keep_pos_tag can be 'all' for no pos_tag else list of tags to keeps.
    #     lemmatize is boolean to know if you wan to apply lemmatization by Spacy model.
    method_embedding: dict = field(
        default_factory=lambda: {'Word2vec': 'Word2Vec',
                                 'Fasttext': 'FastText',
                                 'Doc2Vec': 'Doc2Vec',
                                 'Transformer': 'CamemBERT',
                                 'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False),
                                           (['ADJ', 'NOUN', 'VERB', 'DET'], True)]})
    # if True, apply Hyperparameters Optimization else load json of models parameters from path indicated
    # in flags.path_models_parameters
    apply_optimization: bool = True
    # if True, apply validation / cross-validation and save models
    apply_validation: bool = True
    # if True, apply blend of all NLP models : average of all predictions
    apply_blend_model: bool = False
    # a path of json file indicating models parameters
    path_models_parameters: str = None
    # a path of json file indicating best models parameters
    # if None or didn't find the path, it will search for the file "models_best_parameters.json" in tracking file
    # if fail : create a new json file "models_best_parameters.json" in tracking file
    # warning : must keep same seed and scoring metric for comparaison
    path_models_best_parameters: str = None
    # verbosity levels:`0`: No messages / `1`: Warnings / `2`: Info / `3`: Debug.
    verbose: int = 2

    ### Only for Neural Network NN
    batch_size: int = 16
    # patience for early stopping:
    patience: int = 4
    # instantiate a large number because it uses early stopping :
    epochs: int = 60
    # minimum for learning rate reduction
    min_lr: float = 1e-4

    # Display
    # sort dataframe leaderboard by a metric
    # binary : ['accuracy','f1','recall','precision','roc_auc']
    # classification : ['accuracy','f1','recall','precision']
    # regression : ['mse','explained_variance','r2']
    sort_leaderboard: str = "accuracy"

    # dictionary of map target if target are not numerics
    map_label: dict = field(default_factory=dict)

    ### Embedding
    dimension_embedding = "doc_embedding"

    ### Model parameters
    # TF matrix
    tf_binary: bool = True
    tf_ngram_range: list = field(default_factory=lambda: [(1, 1), (1, 2), (1, 3)])
    tf_stop_words: bool = True

    # TF word matrix
    # use an unique TF matrix to get word vectors in order to use it for embedding
    tf_wde_binary: bool = True
    tf_wde_stop_words: bool = True
    tf_wde_ngram_range: tuple = field(default_factory=lambda: (1, 1))
    tf_wde_vector_size: int = 200
    tf_wde_max_features: int = 20000
    tf_wde_maxlen: int = 250
    tf_wde_learning_rate: float = field(default_factory=lambda: [1e-3])

    # TF-IDF matrix
    tfidf_binary: bool = True
    tfidf_ngram_range: list = field(default_factory=lambda: [(1, 1), (1, 2), (1, 3)])
    tfidf_stop_words: bool = True

    # TF-IDF word matrix
    # use an unique TF-IDF matrix to get word vectors in order to use it for embedding
    tfidf_wde_binary: bool = True
    tfidf_wde_stop_words: bool = True
    tfidf_wde_ngram_range: tuple = field(default_factory=lambda: (1, 1))
    tfidf_wde_vector_size: int = 200
    tfidf_wde_max_features: int = 20000
    tfidf_wde_maxlen: int = 250
    tfidf_wde_learning_rate: float = field(default_factory=lambda: [1e-3])

    # Word2Vec embedding
    # if you want to create a scratch model:
    w2v_size_vector: int = 300
    w2v_window: int = 5
    w2v_epochs: int = 10
    w2v_sg: int = 0
    # Training :
    w2v_maxlen: int = 250
    w2v_max_features: int = 20000
    w2v_learning_rate: list = field(default_factory=lambda: [1e-3])

    # FastText embedding
    # if you want to create a scratch model:
    ft_size_vector: int = 300
    ft_window: int = 5
    ft_epochs: int = 10
    ft_thr_grams: int = 10
    ft_sg: int = 0
    # Training :
    ft_maxlen: int = 250
    ft_max_features: int = 20000
    ft_learning_rate: list = field(default_factory=lambda: [1e-3])

    # Doc2vec embedding
    # if you want to create a scratch model:
    d2v_size_vector: int = 300
    d2v_window: int = 5
    d2v_epochs: int = 10
    d2v_sg: int = 0
    # Training :
    d2v_maxlen: int = 250
    d2v_max_features: int = 20000
    d2v_learning_rate: list = field(default_factory=lambda: [1e-3])

    # Transformer embedding
    tr_maxlen: int = 258
    tr_learning_rate: list = field(default_factory=lambda: [1e-4])

    ### Classifier Sklearn
    # Naive Bayes
    nb_alpha_min: float = 0.0
    nb_alpha_max: float = 1.0

    # Logistic Regression
    logr_C_min: float = 1e-2
    logr_C_max: float = 1e2
    logr_penalty: list = field(default_factory=lambda: ['l2', 'l1'])

    # SGD Classifier or Regressor
    sgd_alpha_min: float = 1e-4
    sgd_alpha_max: float = 1e-2
    sgdc_penalty: list = field(default_factory=lambda: ['l2', 'l1'])
    # don't use 'hinge' (can't use predict_proba):
    sgdc_loss: list = field(default_factory=lambda: ['log', 'modified_huber'])
    sgdr_penalty: list = field(default_factory=lambda: ['l2', 'l1'])
    sgdr_loss: list = field(default_factory=lambda: ['squared_loss', 'huber', 'epsilon_insensitive'])

    # Random Forest Classifier or Regressor
    rf_n_estimators_min: int = 20
    rf_n_estimators_max: int = 100
    rf_max_depth_min: int = 5
    rf_max_depth_max: int = 75
    rf_min_samples_split_min: int = 5
    rf_min_samples_split_max: int = 15
    rf_max_samples_min: float = 0.5
    rf_max_samples_max: float = 1

    # LightBGM Classifier or Regressor
    lgbm_n_estimators_min: int = 20
    lgbm_n_estimators_max: int = 200
    lgbm_num_leaves_min: int = 5
    lgbm_num_leaves_max: int = 150
    lgbm_learning_rate_min: float = 0.03
    lgbm_learning_rate_max: float = 0.3
    lgbm_bagging_fraction_min: float = 0.5
    lgbm_bagging_fraction_max: float = 1

    # XGBoost
    xgb_n_estimators_min: int = 20
    xgb_n_estimators_max: int = 200
    xgb_max_depth_min: int = 3
    xgb_max_depth_max: int = 10
    xgb_learning_rate_min: float = 0.04
    xgb_learning_rate_max: float = 0.3
    xgb_subsample_min: float = 0.5
    xgb_subsample_max: float = 1.0

    # CatBoost
    cat_iterations_min: int = 20
    cat_iterations_max: int = 200
    cat_depth_min: int = 2
    cat_depth_max: int = 9
    cat_learning_rate_min: float = 0.04
    cat_learning_rate_max: float = 0.3
    cat_subsample_min: float = 0.5
    cat_subsample_max: float = 1

    # Dense Neural Network
    dnn_hidden_unit_1_min: int = 60
    dnn_hidden_unit_1_max: int = 120
    dnn_hidden_unit_2_min: int = 60
    dnn_hidden_unit_2_max: int = 120
    dnn_hidden_unit_3_min: int = 60
    dnn_hidden_unit_3_max: int = 120
    dnn_learning_rate: list = field(default_factory=lambda: [1e-2, 1e-3])
    dnn_dropout_rate_min: float = 0.0
    dnn_dropout_rate_max: float = 0.5

    ### Classifier Neural Network
    # GlobalAverage
    ga_dropout_rate_min: float = 0
    ga_dropout_rate_max: float = 0.5

    # RNN
    rnn_hidden_unit_min: int = 120
    rnn_hidden_unit_max: int = 130
    rnn_dropout_rate_min: float = 0
    rnn_dropout_rate_max: float = 0.5

    # LSTM
    lstm_hidden_unit_min: int = 120
    lstm_hidden_unit_max: int = 130
    lstm_dropout_rate_min: float = 0
    lstm_dropout_rate_max: float = 0.5

    # GRU
    gru_hidden_unit_min: int = 120
    gru_hidden_unit_max: int = 130
    gru_dropout_rate_min: float = 0
    gru_dropout_rate_max: float = 0.5

    # Attention
    att_dropout_rate_min: float = 0
    att_dropout_rate_max: float = 0.5

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        dump(content, f, width=width)


def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = full_load(f)
    return content
