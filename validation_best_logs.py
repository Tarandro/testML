import pandas as pd
from autonlp.autonlp import AutoNLP
import os
from autonlp.flags import Flags, load_yaml

#####################
# Parameters
#####################

path_logs = "./logs"
flags_dict_info = load_yaml(os.path.join(path_logs, "flags.yaml"))
flags = Flags().update(flags_dict_info)

flags_dict_autonlp = {
    "apply_optimization": False,
    "apply_validation": True,
    "path_models_parameters": "./logs/models_best_parameters.json"
}

flags = flags.update(flags_dict_autonlp)
print("flags :", flags)

# FastText work only with pre-training dataset on kaggle (see url method_embedding)
# Need GPU for BERT
# NLP models: ['tf-idf+Naive_Bayes', 'tf-idf+SGD_Classifier', 'tf-idf+Logistic_Regression',
# 'Fasttext+bigru_attention', 'Transformer+global_average']

if __name__ == '__main__':
    #####################
    # data
    #####################

    data = pd.read_csv(flags.path_data)

    autonlp = AutoNLP(flags)

    #####################
    # Preprocessing
    #####################

    autonlp.data_preprocessing(data)

    #####################
    # Training
    #####################

    autonlp.train()

    #####################
    # Ensemble
    #####################

    autonlp.ensemble()

    #####################
    # Leaderboard (Validation score)
    #####################

    leaderboard_val = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='val')
    print('\nValidation Leaderboard')
    print(leaderboard_val)
    #autonlp.save_scores_plot(leaderboard_val, 'last_logs')
    leaderboard_val.to_csv('./results/leaderboard_val.csv', index=False)

    autonlp.correlation_models()

    # autonlp.show_distribution_scores()

    # df_oof_val = autonlp.Y_train.copy()
    # for name in autonlp.models.keys():
    #     df_oof_val[name] = np.argmax(autonlp.models[name].info_scores['oof_val'], axis=1).reshape(-1)
    # df_oof_val.to_csv('./results/results_nlp/df_oof_val.csv', index=False)

    if 'binary' in autonlp.objective:
        autonlp.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    name_logs = 'last_logs'
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data)

    df_prediction = autonlp.dataframe_predictions
    df_prediction.to_csv('./results/df_prediction.csv', index=False)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test')
    print('\nTest Leaderboard')
    print(leaderboard_test)
    #autonlp.save_scores_plot(leaderboard_test, 'last_logs')
    leaderboard_test.to_csv('./results/leaderboard_test.csv', index=False)
