# AutoNLP classes

### class : AutoNLP

AutoNLP is the main base/class: initialization, preprocessing, train / split test, run models, get leaderboard,
correlation of models, confusion matrix and run predictions.

Pour chaque modèle :

* launch an optimization of parameters + retrieves the best parameters
* then validation or cross-validation + model saving at each fold
* possibility of prediction on test set or other data for each model and each fold

### class : Preprocessing_NLP

data preprocessing, first task performed:

1. load NLP spacy model
2. (Optional) small_clean : remove '\n', '\r', URL and numbers
3. (Optional) Train NLP spacy model on text data
4. (Optional) Preprocessing of named entities : Microsoft is replaced by 'ORG', or London by 'LOC'

### class : Model

Model is the parent class of models, in the class is directly implemented:

* fit_optimization
* validation
* prediction
* autonlp

### class_optimization.py : 2 classes

* Optimiz_hyperopt for sklearn models
* Optimiz_hyperopt_NN for neural network models

Using hyperopt: Bayesian TPE optimization

One of the Parameters: search time for each model

Return best parameters for the function to optimize

### class : Validation

validation if validation dataset is provided else apply cross-validation

* Validation / StratifiedKfold or Kfold
* train model
* save model at each fold
* compute score for validation set

### class : Prediction

load models + prediction + compute score if y_test is provided

### class : Flags

list of parameters

* class built with dataclass library
* Possibility to update the parameters
* Saving in yaml format