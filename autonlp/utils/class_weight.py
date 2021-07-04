import numpy as np
from sklearn.utils.class_weight import compute_class_weight


#####################
# Class weight for Neural Network
#####################


def compute_dict_class_weight(y, class_weight, objective):
    """ Compute class weight in a adapted format for Tensorflow model """
    if class_weight == "balanced":
        if ('binary' in objective) or (y.shape[1] == 1 and 'classification' in objective):
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y.reshape(-1)), y=y.reshape(-1))
            return dict(zip(np.unique(y.reshape(-1)), weights))
        else:
            return None
    else:
        return None
