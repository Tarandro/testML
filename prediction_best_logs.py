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
flags.update({"outdir": path_logs})
print("flags :", flags)

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
    # Testing
    #####################

    on_test_data = True
    name_logs = 'best_logs'
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data)

    df_prediction = autonlp.dataframe_predictions

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test', info_models=autonlp.info_models)
    print('\nTest Leaderboard')
    print(leaderboard_test)
