# Preprocessing (cleaner.py)

preprocessing :
* Stop-word (seulement Statistical methods et optionnel)
* Lemmatisation (seulement Statistical methods et optionnel)
* Remplacement des entités nommées (optionnel) : Microsoft est remplacé par 'ORG' ou London par 'CITY'.
* Tokenization
* Padding (pour word/sentence encoders and NLP Transformers)
    
    
Pour word encoders, sentence encoders et NLP Transformers, nous utilisons des modèles pré-entraînés sur des grands
volume de données non pré-traitées. Il n'y a donc pas nécessité de supprimer les Stop-words
ni d'appliquer une technique de stemming ou lemmatisation.
    
De plus, NLP transformers utilise BPE (Byte - Pair Encoding) pour la tokenisation du corpus (running -> run + ##ing),
qui effectue indirectement une lemmatisation. Aussi, NLP transformers utilise le mécanisme de self-Attention,
il donne beaucoup d'importance au mots qui impact la classification, de ce fait les mots à fréquence élevés
tel que les stop-words ont souvent un faible poids et il n'est donc pas nécessaire de les supprimer.