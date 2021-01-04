import shap
import numpy as np 

def _shap_ordering(feature_names, shap_values, task='classification'):
    if task == 'classification' and type(shap_values) == type(list()):
        feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        return feature_names[feature_order][::-1]
    else:
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        return feature_names[feature_order][::-1]
    
    
def shap_select(model, X_train, X_test, feature_names, task='classification', agnostic=False):
    """
    Return the feature ordering of a multidimensional dataset based on the features importance.
    The importance is calculated upon SHAP values, which takes into account a fitted model.


    :param model: a fitted model 
    :param X_train: training data
    :param X_test: test data
    :param feature_names: feature names
    :return: Ordered feature names based on the importance computed using SHAP values
    """
    
    explainer = None
    
    if not agnostic:
        explainer = shap.TreeExplainer(model)
    else:
        background = None
        if len(X) < 500:
            background = X_train
        else:
            background = shap.sample(X_train, int(len(X_train)*0.05))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer.shap_values(X_test)
    ordering = _shap_ordering(feature_names, shap_values)
    
    return ordering