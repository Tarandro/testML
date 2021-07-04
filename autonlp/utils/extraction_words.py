import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras import backend as K
import numpy as np
from IPython.core.display import display, HTML
from collections import Counter
import random as rd
import ast
import copy
from tqdm.notebook import tqdm
from ..utils.logging import get_logger, verbosity_to_loglevel
logger = get_logger(__name__)

##################
# Extraction words
##################
#    pr = {0: 'NEGATIVE', 1: 'POSITIVE'}
#    n_influent_word = 10
#    type_data = 'train'  # 'test'
#
#    html = extract_influent_word(autonlp, type_data, n_influent_word, pr)
#    Html_file = open("./results/results_nlp/extract_word.html", "w")
#    Html_file.write(html)
#    Html_file.close()


def attention_weight(x, fixed_weights_attention, biais_attention, step_dim):
    """ redo the calculations made in the attention layer to obtain the weights """
    """ fixed_weights_attention (array) : Fixed weight of the learned attention layer
        biais_attention (array) : bias of the learned attention layer
        step_dim (int) : maxlen """
    """ return : weights (array)"""

    features_dim = fixed_weights_attention.shape[0]

    eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                          K.reshape(fixed_weights_attention, (features_dim, 1))), (-1, step_dim))

    eij += biais_attention

    eij = K.tanh(eij)

    a = K.exp(eij)

    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

    weights = K.expand_dims(a)
    # weighted_input = x * a
    return weights


def extract_influent_word(X, Y, model_nlp, loaded_models, model_paths, X_is_train_data=False, n_influent_word=10, nb_example='max', pr=None):
    """ extraction of influential words : highlight the words with the highest weight
        for 'fasttext+bigru_attention' and 'transformer+global_average' model
    Args:
        autonlp : class from AutoNLP with 'fasttext+bigru_attention' and 'transformer+global_average' model already trained
        type_data (str) : 'train' or 'test', use documents from train or test dataset
        n_influent_word (int) : number of words to highlight by documents
        nb_example (int or str) : "max" show all data else limit nbr examples
        pr (dict) : (not use for the moment)
    """

    if 'binary_proba' in model_nlp.objective:
        logger.error(" Extraction doesn't work with objective = binary_proba ")
        return None

    if model_nlp.flags_parameters.map_label != dict():
        pr = {v: k for k, v in model_nlp.flags_parameters.map_label.items()}
    else:
        pr = None

    if "attention" in model_nlp.name_classifier.lower():
        name_model_att = True
    else:
        name_model_att = False

    if not model_nlp.is_NN:
        logger.error(" Extraction work only with Neural Network models ")
        return None

    if isinstance(model_nlp.column_text, int) and model_nlp.column_text not in X.columns:
        col = model_nlp.column_text
    else:
        col = list(X.columns).index(model_nlp.column_text)

    dataset = X.iloc[:, [col]].copy().reset_index(drop=True)
    dataset = dataset.sample(frac=model_nlp.flags_parameters.frac_trainset, random_state=model_nlp.seed)
    target = Y.iloc[:, [0]].copy().reset_index(drop=True)
    target = target.sample(frac=model_nlp.flags_parameters.frac_trainset, random_state=model_nlp.seed)

    if model_nlp.embedding.name_model == "transformer":
        x_preprocessed = model_nlp.embedding.preprocessing_transform(dataset)
        ids_d = x_preprocessed[0]
        att_d = x_preprocessed[1]
        tok_d = x_preprocessed[2]

        punctuation_token = []
        for token in [".", ",", "?", ";", "!", "'", ")", "("]:
            enc = model_nlp.embedding.tokenizer.encode("h" + token)
            punctuation_token.append(enc[2])

    else:
        ids_d = model_nlp.embedding.preprocessing_transform(dataset)['tok']

        punctuation_token = []
        for token in [".", ",", "?", ";", "!", "'", ")", "("]:
            enc = model_nlp.embedding.tokenizer.texts_to_sequences([token])
            punctuation_token.append(enc[0])

    if nb_example == "max":
        nb_example = len(dataset)

    rd.seed(model_nlp.seed)
    fold_to_train = rd.sample([i for i in range(model_nlp.flags_parameters.nfolds)],
                              k=max(min(model_nlp.flags_parameters.nfolds_train, model_nlp.flags_parameters.nfolds), 1))
    if model_nlp.flags_parameters.cv_strategy == "StratifiedKFold":
        skf = StratifiedKFold(n_splits=model_nlp.flags_parameters.nfolds, random_state=model_nlp.seed, shuffle=True)
        folds = skf.split(target, target)
    else:
        kf = KFold(n_splits=model_nlp.flags_parameters.nfolds, shuffle=True, random_state=model_nlp.seed)
        folds = kf.split(target)

    html = ''

    dict_loaded_models = {}
    for j, path in enumerate(model_paths):
        for i in range(model_nlp.flags_parameters.nfolds):
            if "fold" + str(i) in path:
                dict_loaded_models[j] = i

    dict_result = {k: None for k in range(len(dataset))}

    for num_fold, (train_index, val_index) in enumerate(folds):
        if num_fold not in fold_to_train:
            continue

        model = loaded_models[dict_loaded_models[num_fold]]

        # tokenize for FastText
        if name_model_att:
            fixed_weights_attention = model.layers[-2].get_weights()[0]
            features_dim = fixed_weights_attention.shape[0]
            biais_attention = model.layers[-2].get_weights()[1]

            # Extraction Model (outputs layer outputs from autonlp.models[name_model_ft].best_model)
            extract_model_attention = tf.keras.Model(inputs=model.input,
                                                              outputs=(
                                                              model.layers[-4].output, model.layers[-1].output))

        else:
            all_layer_weights_transformer = model.layers[-1].get_weights()[0]
            # Extraction Model (outputs layer outputs from autonlp.models[name_model_tr].best_model)
            extract_model_normal = tf.keras.Model(inputs=model.input,
                                                       outputs=(model.layers[-4].output, model.layers[-1].output))

        if not X_is_train_data:
            val_index = [i for i in range(len(ids_d))]

        for k in tqdm(val_index):

            # if len(dataset.iloc[k,:]) > 500: continue  # trop large à afficher

            if name_model_att:
                # USE EXTRACT MODEL
                if model_nlp.embedding.name_model == "transformer":
                    embedding_output, pred_vec = extract_model_attention.predict([ids_d[k:k + 1, :], att_d[k:k + 1, :], tok_d[k:k + 1, :]])
                else:
                    embedding_output, pred_vec = extract_model_attention.predict([ids_d[k:k + 1]])
                embedding_output = np.squeeze(embedding_output[0])  # dim (MAX_LEN,256)
                pred = int(np.argmax(pred_vec))
                weights_attention = attention_weight(embedding_output, fixed_weights_attention, biais_attention,
                                                     model_nlp.embedding.maxlen)
                weights_attention = np.squeeze(weights_attention[0])

                indexes = np.isin(ids_d[k], weights_attention)
                weights_attention = np.where(indexes, np.zeros(weights_attention.shape), weights_attention)

                if dict_result[dataset.index[k]] is None:
                    dict_result[dataset.index[k]] = (k, target.iloc[k, 0], pred, ids_d[k], weights_attention)
                else:
                    previous_result = dict_result[dataset.index[k]]
                    if isinstance(previous_result[2], int):
                        list_pred = [previous_result[2]]
                    else:
                        list_pred = previous_result[2]
                    dict_result[dataset.index[k]] = (k, target.iloc[k, 0], list_pred + [pred], ids_d[k], previous_result[4] + weights_attention)

            else:
                # USE EXTRACT MODEL
                if model_nlp.embedding.name_model == "transformer":
                    embedding_output_3, pred_vec_3 = extract_model_normal.predict([ids_d[k:k + 1, :], att_d[k:k + 1, :], tok_d[k:k + 1, :]])
                else:
                    embedding_output_3, pred_vec_3 = extract_model_normal.predict([ids_d[k:k + 1]])
                embedding_output_3 = np.squeeze(embedding_output_3[0])  # dim (MAX_LEN,768)
                pred_3 = int(np.argmax(pred_vec_3))
                layer_weights_3 = all_layer_weights_transformer[:, pred_3]
                final_output_3 = np.dot(embedding_output_3, layer_weights_3)

                indexes = np.isin(ids_d[k], punctuation_token)
                final_output_3 = np.where(indexes, np.zeros(final_output_3.shape), final_output_3)

                if dict_result[dataset.index[k]] is None:
                    dict_result[dataset.index[k]] = (k, target.iloc[k, 0], pred_3, ids_d[k], final_output_3)
                else:
                    previous_result = dict_result[dataset.index[k]]
                    if isinstance(previous_result[2], int):
                        list_pred = [previous_result[2]]
                    else:
                        list_pred = previous_result[2]
                    dict_result[dataset.index[k]] = (k, target.iloc[k, 0], list_pred + [pred_3], ids_d[k], previous_result[4] + final_output_3)

    # Ensemble :
    for key, value in dict_result.items():
        if isinstance(value[2], list):
            weights = value[4]/len(value[2])
            x = Counter(value[2])
            vote_pred = x.most_common(1)[0][0]
            dict_result[key] = (value[0], value[1], vote_pred, value[3], weights)

    for idx_k in tqdm(range(nb_example)):

        if dict_result[idx_k] is None: continue

        k, val_true, pred, ids, weights_attention = dict_result[idx_k][0], dict_result[idx_k][1], dict_result[idx_k][
                2], dict_result[idx_k][3], dict_result[idx_k][4]

        if pred == val_true:
            success = "Prediction True"
        else:
            success = "Prediction False"

        # DISPLAY TEXT
        # html = ''
        if pr is not None:
            info = 'Train row %i. Predict %s.   True label is %s. %s' % (idx_k, pr[pred], pr[val_true], success)
        else:
            info = 'Train row %i. Predict %s.   True label is %s. %s' % (idx_k, pred, val_true, success)
        html += info + '<br><br>'

        if model_nlp.embedding.name_model == "transformer":
            idx = np.sum(att_d[k,])

            if idx < n_influent_word * 2:
                n_influent_word__ = int(idx / 2)
            else:
                n_influent_word__ = n_influent_word
        else:
            idx = list(ids).count(0)
            nb_tok = model_nlp.embedding.maxlen - idx
            if nb_tok < n_influent_word * 2:
                n_influent_word__ = int(nb_tok / 2)
            else:
                n_influent_word__ = n_influent_word

        if name_model_att:
            weights_attention = weights_attention[:-idx]
            v = np.argsort(weights_attention)
            mx = weights_attention[v[-1]]
            x = max(-n_influent_word, -len(v))
            mn = weights_attention[v[x]]

        else:
            ## technique 3
            v = np.argsort(weights_attention[:idx - 1])
            mx = weights_attention[v[-1]]
            x = max(-n_influent_word, -len(v))
            mn = weights_attention[v[x]]

        html += '<b>' + model_nlp.name_model_full + ' &emsp;&nbsp;:</b>'

        if model_nlp.embedding.name_model == "transformer":
            tokenize = model_nlp.embedding.tokenizer.tokenize(model_nlp.embedding.tokenizer.decode(ids))
            list_ = []
            if True:  # pred_3 == target[k]
                for j in range(1, idx):
                    x = (weights_attention[j] - mn) / (mx - mn)
                    list_.append(x)
                g = list(np.argsort(list_))[::-1]
                for j in range(1, idx):
                    if j - 1 in g[:n_influent_word__]:
                        x = 1 - g.index(j - 1) * 0.7 / n_influent_word
                    else:
                        x = 0
                    try:
                        if tokenize[j][0] == '▁':
                            html += ' '
                    except:
                        pass
                    html += "<span style='background:{};font-family:monospace'>".format('rgba(255,255,0,%f)' % x)
                    html += model_nlp.embedding.tokenizer.decode([ids[j]])
                    html += "</span>"
            html += "<br>"

        else:
            list_ = []
            for j in range(len(weights_attention)):
                x = (weights_attention[j] - mn) / (mx - mn)
                list_.append(x)
            g = list(np.argsort(list_))[::-1]
            for j in range(len(weights_attention)):
                if j in g[:n_influent_word__]:
                    x = 1 - g.index(j) * 0.7 / n_influent_word
                else:
                    x = 0

                html += ' '
                html += "<span style='background:{};font-family:monospace'>".format('rgba(255,255,0,%f)' % x)
                html += model_nlp.embedding.tokenizer.sequences_to_texts([ids[j:j + 1]])[0]
                html += "</span>"
            html += "<br>"

        html += '<br><br><br>'
    display(HTML(html))
    return html, dict_result


def get_top_influent_word(dict_result, model_nlp, n_gram=2, top_k=20, min_threshold=10):
    all_preds = [info[2] for info in dict_result.values()]
    if model_nlp.flags_parameters.map_label != dict():
        pr = {v: k for k, v in model_nlp.flags_parameters.map_label.items()}
    else:
        pr = None
    dict_cluster = {i: {} for i in np.unique(all_preds)}

    if model_nlp.embedding.name_model == "transformer":
        if model_nlp.embedding.method_embedding.lower() in ['roberta', "camembert", "xlm-roberta"]:
            special_token = [0, 1, 5, 6]
        else:
            special_token = [0, 101, 102]
    else:
        special_token = [0]

    html = ''

    for k in dict_result.keys():

        if dict_result[k] is None: continue

        k, val_true, pred, ids, weights_attention = dict_result[k][0], dict_result[k][1], dict_result[k][2], \
                                                    dict_result[k][3], dict_result[k][4]

        # sum of weights of sequences of k words :
        for i in range(model_nlp.embedding.maxlen - n_gram):
            liste_k_words = list(ids[i:i + n_gram])
            liste_k_weights = list(weights_attention[i:i + n_gram])
            if all([False if j in special_token else True for j in liste_k_words]):
                liste_k_words = str(liste_k_words)
                if liste_k_words in dict_cluster[pred].keys():
                    dict_cluster[pred][liste_k_words] = (
                        dict_cluster[pred][liste_k_words][0] + np.mean(liste_k_weights),
                        dict_cluster[pred][liste_k_words][1] + 1)
                else:
                    dict_cluster[pred][liste_k_words] = (np.mean(liste_k_weights), 1)

    dict_cluster_copy = copy.deepcopy(dict_cluster)
    # divide the sum of the weights by the number of times the sequence occurs
    for i in dict_cluster_copy.keys():
        for liste_k_words in dict_cluster_copy[i].keys():
            n = dict_cluster_copy[i][liste_k_words][1]
            if n > min_threshold:
                dict_cluster_copy[i][liste_k_words] = dict_cluster_copy[i][liste_k_words][0] / n
            else:
                dict_cluster_copy[i][liste_k_words] = 0

        dict_cluster_copy[i] = {k: v for k, v in
                                sorted(dict_cluster_copy[i].items(), key=lambda item: item[1],
                                       reverse=True)}

    html += '<b>' + 'Top {} terms with the highest average weight :\n'.format(top_k) + '</b>' + '<br><br>'
    for i in dict_cluster_copy.keys():
        if pr is not None:
            html += 'class : <b> {} </b>'.format(pr[i]) + '<br><br>'
        else:
            html += 'class : <b> {} </b>'.format(i) + '<br><br>'
        print()
        sorted_weight_dict = dict_cluster_copy[i]
        terms_sorted = [ast.literal_eval(list_terms) for list_terms in sorted_weight_dict.keys()]
        weight_sorted = [weight for weight in sorted_weight_dict.values()]
        message = ''
        for j in range(min(top_k, len(terms_sorted))):
            if model_nlp.embedding.name_model == "transformer":
                message += model_nlp.embedding.tokenizer.decode(terms_sorted[j])
            else:
                message += model_nlp.embedding.tokenizer.sequences_to_texts(np.array([terms_sorted[j]]))[0]
            message += ' - '
        html += message[:-2] + '<br><br>'

    display(HTML(html))
    return html