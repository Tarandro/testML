# Embeddings and NLP Models

## Classifiers

* Vecteurs des documents comme entrée : (appelé 'document_embedding' dans le projet)

    - Naive Bayes
    - Régression Logistique
    - Régression linéaire
    - XGBoost
    
* Vecteurs des mots des documents comme entrée : (appelé 'word_embedding' dans le projet)

    - Global Average    
    - Attention
    - RNN / BiRNN / BiRNN + Attention
    - LSTM / BiLSTM / BiLSTM + Attention
    - GRU / BiGRU / BiGRU + Attention 
    
Les méthodes de vectorisation de textes permettent d’obtenir une représentation vectorielle pour chaque mot.
Pour construire l’entrée d’un modèle de classification, la première idée serait de sommer les vecteurs des mots
pour obtenir un seul vecteur représentant le document. Cependant, les prépositions ou déterminants peuvent prendre
le dessus dans l’addition et donc cacher les mots qui pourraient avoir une importance dans la classification.
Les mécanismes tel que RNN, LSTM, GRU ou Attention ont pour but de montrer où est l’information pertinente.

__BiGRU :__ Bi-directionnal Gated Recurrent unit, est une couche bi-directionnelle de type RNN. GRU est très similaire
à la couche LSTM avec moins de paramètres et possède seulement 2 fonctions d'entrées. L'une 'update gate'
décide des informations à garder et des informations à ajouter provenant de la variable d'entrée et
l'autre 'reset gate' décide de combien d'informations vont être supprimées. Le mécanisme BiGRU montre où est
l’information pertinente en ne gardant à la sortie de la couche que les informations concernant les mots
les plus influents pour la classification.

__Attention :__ Le mécanisme Attention a pour but de montrer où est l’information pertinente en ajoutant un poids
à chaque mot du document.