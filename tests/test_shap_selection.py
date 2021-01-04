import catboost as cb 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_boston, load_diabetes

from shap_selection import feature_selection


def test_shap_select():

	iris_data = load_iris()

	X, y = iris_data.data, iris_data.target
	feature_names = np.array(iris_data.feature_names)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = cb.CatBoostClassifier(verbose=False)    
	model.fit(X_train, y_train)

	feature_order = feature_selection.shap_select(model, X_train, X_test, feature_names, agnostic=False)

	assert len(feature_order) == len(feature_names)

	wine_data = load_wine()

	X, y = wine_data.data, wine_data.target
	feature_names = np.array(wine_data.feature_names)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = cb.CatBoostClassifier(verbose=False)    
	model.fit(X_train, y_train)

	feature_order = feature_selection.shap_select(model, X_train, X_test, feature_names, agnostic=False)

	assert len(feature_order) == len(feature_names)