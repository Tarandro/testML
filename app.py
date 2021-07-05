import pandas as pd
from autonlp.autonlp import AutoNLP
import os
from autonlp.flags import Flags, load_yaml

from fastapi import FastAPI
from fastapi import File, UploadFile, Body
import autonlp.autonlp as base
import autonlp.models.classifier_nlp.trainer as nlp
import pandas as pd
import numpy as np
from fastapi import HTTPException
from typing import Optional
import os

# uvicorn app:app --port 5000

app = FastAPI(title="AutoNLP API", description="API for NLP dataset", version="1.0")


#####################
# Parameters
#####################


@app.on_event('startup')
async def load_flags():
    path_model_deployment = "./model_deployment"
    flags_dict_info = load_yaml(os.path.join(path_model_deployment, "flags.yaml"))
    flags = Flags().update(flags_dict_info)
    flags.update({"apply_app": True})

    base.autonlp = AutoNLP(flags)


#####################
# Load Model at the start
#####################


@app.on_event('startup')
async def load_model():
    nlp.model_nlp, nlp.loaded_models = base.autonlp.create_model_class()


#####################
# Predictions
#####################


@app.post('/predict', tags=["predictions"])
async def get_prediction(text: Optional[str] = "Le marché Français a diminué de 50%",
                         csv_file: UploadFile = File(None, title="data")):
    if csv_file is not None:
        data = pd.read_csv(csv_file.file)

        preprocessed_value = base.autonlp.preprocess_test_data(data)
        if len(preprocessed_value) == 2:
            data_test, doc_spacy_data_test = preprocessed_value
            y_test = None
        else:
            data_test, doc_spacy_data_test, y_test = preprocessed_value
    else:
        data_test, doc_spacy_data_test = base.autonlp.preprocess_test_data(text)
        y_test = None

    nlp.model_nlp.column_text = base.autonlp.column_text

    # y=data[base.autonlp.target]
    dict_prediction, scores = base.autonlp.single_prediction(x=data_test, y=y_test, proba=False,
                                                        model_ml=nlp.model_nlp, loaded_models=nlp.loaded_models,
                                                        return_scores=True)

    dict_prediction = {"prediction": dict_prediction["prediction"].reshape(-1).tolist(),
                       "confidence": np.round(dict_prediction["confidence"].reshape(-1), 3).tolist()}
    if base.autonlp.flags_parameters.map_label != {}:
        map_label = base.autonlp.flags_parameters.map_label
        reverse_label = {v: k for k, v in map_label.items()}
        dict_prediction["prediction"] = [reverse_label[label] for label in dict_prediction["prediction"]]
    if scores is not None:
        dict_prediction["scores"] = scores
    return dict_prediction


@app.post('/predict_proba', tags=["predictions"])
async def get_prediction(text: Optional[str] = "Le marché Français a diminué de 50%",
                         csv_file: UploadFile = File(None, title="data")):
    if csv_file is not None:
        data = pd.read_csv(csv_file.file)

        preprocessed_value = base.autonlp.preprocess_test_data(data)
        if len(preprocessed_value) == 2:
            data_test, doc_spacy_data_test = preprocessed_value
            y_test = None
        else:
            data_test, doc_spacy_data_test, y_test = preprocessed_value
    else:
        data_test, doc_spacy_data_test = base.autonlp.preprocess_test_data(text)
        y_test = None

    nlp.model_nlp.column_text = base.autonlp.column_text

    # y=data[base.autonlp.target]
    prediction, scores = base.autonlp.single_prediction(x=data_test, y=y_test,
                                                        model_ml=nlp.model_nlp, loaded_models=nlp.loaded_models,
                                                        return_scores=True)

    dict_prediction = {}
    if base.autonlp.flags_parameters.map_label != {}:
        dict_prediction["map_label"] = base.autonlp.flags_parameters.map_label
    if scores is not None:
        dict_prediction["scores"] = scores
    if len(prediction.shape) == 1:
        dict_prediction["prediction_1"] = np.round(prediction.reshape(-1), 3).tolist()
    else:
        for i in range(prediction.shape[1]):
            dict_prediction[i] = np.round(prediction[:, i].reshape(-1), 3).tolist()
    return dict_prediction
