# SHAP-Selection: Selecting feature using SHAP values

Due to the increasing concerns about machine learning interpretability, we believe that interpretation could be added to pre-processing steps. Using this library, you will be able to select the most important features from a multidimensional dataset while being able to explain your decisions!

To use SHAP-Selection, you will need:
  * [SHAP](https://github.com/slundberg/shap)

## Instalation
```
pip install shap-selection
```

## Citation

```BibTex
@INPROCEEDINGS{MarcilioJr2020shapselection,  
  author={W. E. {Marcílio} and D. M. {Eler}}, 
  booktitle={2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},   
  title={From explanations to feature selection: assessing SHAP values as feature selection mechanism},   
  year={2020},  
  pages={340-347},  
  doi={10.1109/SIBGRAPI51738.2020.00053}
}
```

## Usage 

To use SHAP-Selection, you must have a trained model. It works both for classification and regression purposes!

##### Load a dataset

```python
iris_data = load_iris()

X, y = iris_data.data, iris_data.target
feature_names = np.array(iris_data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

##### Fit a model

```python
model = cb.CatBoostClassifier(verbose=False)    
model.fit(X_train, y_train)
```

##### Use SHAP-Selection

```python

from shap_selection import feature_selection

# please, use agnostic = True to use with any model...
# agnostic = True will only work with tree-based models
feature_order = feature_selection.shap_select(model, X_train, X_test, feature_names, agnostic=False)
```

### Support 

Please, if you have any questions feel free to contact me at wilson_jr@outlook.com
