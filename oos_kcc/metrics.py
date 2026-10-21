import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def clustering_accuracy(y_true, y_pred):
    a, ia = np.unique(y_true, return_inverse=True)
    b, ib = np.unique(y_pred, return_inverse=True)
    cm = np.zeros((len(a), len(b)), dtype=np.int64)
    np.add.at(cm, (ia, ib), 1)
    r, c = linear_sum_assignment(cm.max() - cm)
    return cm[r, c].sum() / len(y_true)

def all_metrics(y_true, y_pred):
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }
