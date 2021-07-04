# Embeddings and NLP Models

## Embedding methods

### - Statistical features :

__Matrice TF:__

Le modèle Bag-of-word est une manière de représenter un corpus de documents sous forme matricielle, documents x mots du
vocabulaire. La représentation la plus simple est la matrice TF représentant la fréquence des termes dans les documents:

    TF(t,d) = log( 1 + (nombre de fois le terme t apparaît dans le document d)/(nombre total de termes dans le document d))

Cette méthode agit donc comme une extraction de variables à partir d'un vocabulaire de mots. Un des inconvénients de la
méthode est que c’est les termes très fréquents tels que les stop-words qui possèdent les valeurs les plus élevés et
donc cache les mots qui ont plus d'importance dans la classification. Pour y remédier, on peut utiliser la matrice TF-IDF.

__Matrice TF-IDF:__

La matrice TF-IDF est une matrice non négative définie pour un terme t dans un document d défini par :

    TF-IDF(t,d) = TF(t,d) * IDF(t)    où IDF(t) = log(nombre total de document dans le corpus / nombre de documents contenant le terme t)

IDF = inverse document frequency, représente une pondération pour donner plus d'importance aux termes apparaîssant
dans peu de documents

Un inconvénient est la perte du contexte de la phrase car on ne possède plus que la fréquence des mots comme information.
L'utilisation de bi-grammes et tri-grammes peut remédier à ce problème car il est possible leur extraire un contexte.
Pour autant, l'ensemble des n-grams d'une phrase ne permettent pas de définir le contexte de la phrase.

La matrice TF ou TF-IDF obtenue peut ensuite être utilisée comme variable d’entrée à des modèles de classifications
adaptés à la distribution des données textuelles : Naive Bayes classification, régression linéaire, XGBoost ou bien SVM.

### - Word encoders :

Des modèles de Deep Learning ont été créés pour capturer les relations sémantiques des termes et vectoriser des termes
respectant une notion de similarité. C’est un point qu’on ne retrouve pas avec la matrice TF-IDF
qui se repose essentiellement sur la fréquence des mots.

C'est-à-dire que les mots sont transformés en vecteur de dimension spécifique par un processus de word embedding
(il existe différents modèles). Les mots qui apparaissant dans un même contexte sont proches sémantiquement
et ont donc des vecteurs similaires.

__Word2Vec (2013):__

Le modèle Word2Vec est basé sur un réseau de neurones permettant de former des vecteurs pour des mots.
L’approche utilisée est basée sur le modèle skip-gram, pour chaque mot, le modèle prédit le contexte
ou plus précisément les mots entourant le mot en entrée.

L’apprentissage du modèle apprend sous une forme de couche cachée une matrice de dimension V x H avec H la dimension
de la couche cachée et V la taille du vocabulaire du corpus (nombre de mots uniques). Le résultat de cette matrice
permet de donner un vecteur de taille H à chaque mot. La représentation d’un mot après entraînement est unique
et ne dépend pas du contexte de la phrase.

Un inconvénient du modèle est qu'on obtient deux vecteurs différents pour 2 mots à lemme identique
tel que 'manger' et 'mangé' dont la similarité entre ces deux mots peut être faible s'ils sont peu représentés
dans le corpus d'entraînement, le modèle n'aura pas réussi à détecter leur similarité.
Un autre inconvénient est qu'il ne peut produire de vecteur pour un mot non compris dans le corpus d'entraînement,
ce qui peut impacter la prédiction sur des données externes.

__FastText (2016):__

Le modèle FastText est basé sur un réseau de neurones permettant de former des vecteurs pour des lettres n-grams.
Un mot est représenté par un ensemble de lettres n-grams. L’approche est basée sur le modèle skip-gram,
pour chaque lettre n-grams, le modèle prédit les lettres n-grams entourant la lettre n-grams.

L’apprentissage du modèle est identique à word2vec avec pour entrer des lettres n-grams à la place de mots.
L'avantage par rapport à word2vec est qu'on peut former des vecteurs pour des mots non compris dans le corpus d'entraînement.

### - Sentence encoders :

__Doc2Vec (2014):__

Le modèle Doc2Vec est une extension du modèle Word2Vec. Pour le modèle Word2Vec, nous utilisons les vecteurs des mots
pour la prédiction d’autres mots. Doc2Vec rajoute en entrée le vecteur du document, ceci agit une comme un topic du
document aidant à la prédiction et permet d’avoir une représentation numérique du document généralement
plus fiable que la moyenne des vecteurs des mots du document.

### - NLP Transformer :

L'inconvénient des modèles Word encoders est qu'on obtient un vecteur unique pour chaque mot. De ce fait on obtient
un vecteur identique même si le contexte de la phrase est différent. Avec l'ajout de décodeurs et du mécanisme
self-Attention à la place des LSTM/GRU, les modèles Transformers permet d'obtenir une représentation vectorielle
pour un même mot différent selon le contexte de la phrase

Dans les modèles Transformer, l'encodeur est constitué d’une couche Multi-Head Attention et d’un réseau de neurones
à propagation avant. Le décodeur est constitué de deux couches Multi-Head Attention et d’un réseau de neurones à
propagation avant. Le mécanisme Attention a pour objectif d’apporter une importance des mots suivant la tâche,
en ajoutant différent poids aux vecteurs.

Les modèles utilisés ont d'abord été pré-entraîné sur un grand volume de données de façon non-supervisé.
Cette tâche est très longue et coûteuse. On peut ensuite utiliser les poids obtenus après pré-entraînement
pour des tâches supervisées. Ici, nous utilisons les vecteurs obtenus par la dernière couche pour le premier terme,
à savoir [CLS], comme les représentations vectorielles des mots du document.
Ensuite nous appliquons la moyenne de ces vecteurs et couche dense pour classifier les documents.

Les différents modèles Transformers sont différents par rapport aux objectifs lors du pré-entraînement
ou par rapport aux nombres de paramètres utilisés.

Les modèles Transformers utilisent des couches de self-Attention, voir http://jalammar.github.io/illustrated-transformer/

__BERT (2018):__

Bidirectional Encoder Representations from Transformers

Lors du pré-entraînement, la tâche du modèle, sous forme semi-supervisé, consiste à prédire des mots d'une phrase
qui ont été masqués en entrée (15\% des mots) et de prédire si une autre phrase est la phrase suivante.
De ce fait, certains mots auront plus d’influences que d’autres pour prédire les mots masqués
ou de relier les deux phrases d’où l’utilisation du mécanisme Attention.

Pour prédire les mots masqués, BERT s'appuie sur un modèle autorégressif et l'objectif est de reconstruire le mot
masqué en fonction de la phrase avec le mot masqué. Aussi, BERT affirme que tous les mots masqués sont mutuellement indépendants.

__FlauBERT:__

Le modèle FlauBERT repose sur l’architecture du modèle BERT pré-entraîné sur le corpus français réunissant
des articles de CNRS (French National Centre for Scientific Research), CommonCrawl, NewsCrawl, Wikipedia.

__RoBERTa (2019):__

Le modèle RoBERTa est pré-entraîné sur un jeu de données plus conséquent que celui de BERT et la tâche de prédiction
de la phrase suivante est retirée. Il y a donc que la tâche de prédiction des mots masqués.

__CamemBERT:__

Le modèle CamemBERT repose sur l’architecture du modèle RoBERTa pré-entraîné sur le corpus français de OSCAR.

__XLM-RoBERTa:__

XLM pour Cross-lingual Language Model et représente donc un modèle multilingue.

Il s'agit d'un modèle RoBERTa pré-entraîné sur un corpus multi-langues (+100 langues, y compris le français).
Ce qui permet d'obtenir des performances raisonnables même en changeant de langue.