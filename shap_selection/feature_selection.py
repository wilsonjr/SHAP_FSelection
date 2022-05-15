import shap
import numpy as np 

def _shap_ordering(feature_names, shap_values, task='classification'):
    if task == 'classification' and type(shap_values) == type(list()):

        aggreated_values = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)

        feature_order = np.argsort(aggreated_values)
        return feature_names[feature_order][::-1], np.sort(aggreated_values)[::-1]
    else:

        aggreated_values = np.sum(np.abs(shap_values), axis=0)

        feature_order = np.argsort(aggreated_values)


        return feature_names[feature_order][::-1], np.sort(aggreated_values)[::-1]
    
    
def shap_select(model, X_train, X_test, feature_names, task='classification', agnostic=False, background_size=0.1):
    """
    Return the feature ordering of a multidimensional dataset based on the features importance.
    The importance is calculated upon SHAP values, which takes into account a fitted model.


    :param model: a fitted model 
    :param X_train: training data
    :param X_test: test data
    :param feature_names: feature names
    :param task: classification or regression
    :param agnostic: whether to use or not agnostic explanation
    :param background_size: percentage of the datapoints to use as background data
    :return: Ordered feature names based on the importance computed using SHAP values and
             the importance value associated to the features

    """
    
    explainer = None
    
    if not agnostic:
        explainer = shap.TreeExplainer(model)
    else:
        background = None
        if len(X_train) < 500:
            background = X_train
        else:
            background = shap.sample(X_train, int(len(X_train)*background_size))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer.shap_values(X_test)
    ordering, importance_values = _shap_ordering(feature_names, shap_values, task)
    
    return ordering, importance_values