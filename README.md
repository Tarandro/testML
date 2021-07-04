# AutoNLP

AutoNLP is an automated training and deployment of NLP models

## Requirements

Python 3.8 or later with all requirements.txt dependencies installed. To install run:
```python
$ pip install -r requirements.txt
```

## Minimum codes

Setting up:

```python
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags

parameters_to_update = {
    "path_data": "data/FinancialPhraseBank.csv",
    "column_text": "text_fr",
    "target": "sentiment",
    "language_text": "fr",
    "objective": 'multi-class',
    "include_model": ['tf-idf+Naive_bayes', 'tf-idf+SGD_Classifier'],
    "scoring": 'f1',
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'CamemBERT',
                         'spacy': [('all', False)]}
}
flags = Flags().update(parameters_to_update)
autonlp = AutoNLP(flags)
```
Preprocessing, split train/test and Training + Validation:
```python
autonlp.data_preprocessing()
autonlp.train()
leaderboard_val = autonlp.get_leaderboard(dataset='val')
```
Prediction on test set for all models:
```python
autonlp.leader_predict()
df_prediction = autonlp.dataframe_predictions
leaderboard_test = autonlp.get_leaderboard(dataset='test')
```

## Usage examples

To find out how to work with AutoNLP:

- [Tutorial 1: autonlp-caradisiac.ipynb](notebooks/autonlp-caradisiac.ipynb) pipeline using Hyperparameters
  Optimization and validation for models : 'tf-idf+SGD_Regressor', 'tf-idf+xgboost', 'tf-idf+global_average',
  'fasttext+attention' and use manual logs option.
  
- [Tutorial 2: autonlp-caradisiac-mlflow.ipynb](notebooks/autonlp-caradisiac-mlflow.ipynb) pipeline using MLflow Tracking,
  saved in "./mlruns", to open dashboard : $ mlflow ui --backend-store-uri ./mlruns

If you want to apply AutoNLP in several steps, each step can be used multiple times: (manual logs option)

1. [autonlp-caradisiac-optimization.ipynb](notebooks/autonlp-caradisiac-optimization.ipynb) apply only hyperparameters
   optimization, keep a history of each model tested (hyperparameters + score) and save in a json file
   "models_best_parameters.json" hyperparameters with best score for each model.
   
2. [autonlp-caradisiac-validation.ipynb](notebooks/autonlp-caradisiac-validation.ipynb) use previous logs,
   apply validation or cross-validation, save model and compute metric scores.
   
3. [autonlp-caradisiac-prediction.ipynb](notebooks/autonlp-caradisiac-prediction.ipynb) apply prediction on all saved models.

## Parameters - Quick overview

- objective : specify target objective
```python
list_possible_objective = ['binary', 'multi-class', 'regression']
```
    For 'binary', only labels 0 and 1 are possible.
    For 'multi-class', if labels are numerics, labels must be in the range 0 to the number of labels.
- include_model : models (format = "name_embedding+name_classifier") to include in AutoNLP
```python
name_embeddings = ['tf', 'tf-idf', 'word2vec', 'fasttext', 'doc2vec', 'transformer']
name_classifiers = ['naive_bayes', 'logistic_regression', 'sgd_classifier', 'sgd_regressor', 'xgboost',
                    'global_average', 'attention', 'birnn', 'birnn_attention', 'bilstm',
                    'bilstm_attention', 'bigru', 'bigru_attention']
list_possible_models = ['tf+naive_bayes', 'tf-idf+attention', 'transformer+global_average']
```
- method_embedding : information about embedding method
```python
# default:
method_embedding = {'Word2Vec': 'Word2Vec',
                    'FastText': 'FastText',
                    'Doc2Vec': 'Doc2Vec',
                    'Transformer': 'CamemBERT',
                    'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False),
                              (['ADJ', 'NOUN', 'VERB', 'DET'], True)]}
```
    For 'Word2Vec', 'FastText' and 'Doc2Vec', you can create a gensim model from scratch by writing
    the name of the model as embedding method or you can use pre-trained model/weights by indicating the path.
    
    For 'Transformer', you have the choice between these pre-trained models : 'BERT', 'RoBERTa',
    'CamemBERT', 'FlauBERT' and 'XLM-RoBERTa'
    
    For 'Spacy', it doesn't indicate an embedding method but the preprocessing step for
    tf and tf-idf embedding method. You can choose several preprocessing methods in the tuple
    format (keep_pos_tag, lemmatize). keep_pos_tag can be 'all' for no pos_tag else list of tags to keeps.
    lemmatize is boolean to know if apply lemmatization by Spacy model.

- path_data_validation : write a path to use your own validation set instead of using cross-validation
- apply_logs : if True, use a manual logs to track and save model
- apply_mlflow : if True, use MLflow Tracking
- scoring : metric optimized during optimization
```python
binary_posssible_scoring = ['accuracy','f1','recall','precision','roc_auc']
multi_class_posssible_scoring = ['accuracy','f1','recall','precision']
regression_posssible_scoring = ['mse','explained_variance','r2']
```
- apply_optimization : if True, apply Hyperparameters Optimization else load models parameters from path
indicated in flags.path_models_parameters
- apply_validation : if True, apply validation / cross-validation and save models

## Documentation

- [Preprocessing](autonlp/features/README.md)
- [Embeddings](autonlp/features/embeddings/README.md)
- [Classifiers](autonlp/models/classifier/README.md)
- [UML](AutoNLP%20UML.png)
- [Classes](autonlp/README.md)