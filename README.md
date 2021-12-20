.. -*- mode: rst -*-

|pypi_version|_ |pypi_downloads|_

.. |pypi_version| image:: https://img.shields.io/pypi/v/shap-selection.svg
.. _pypi_version: https://pypi.python.org/pypi/shap-selection/

.. |pypi_downloads| image:: https://pepy.tech/badge/shap-selection/month
.. _pypi_downloads: https://pepy.tech/project/shap-selection

=====
SHAP-Selection: Selecting feature using SHAP values
=====

Due to the increasing concerns about machine learning interpretability, we believe that interpretation could be added to pre-processing steps. Using this library, you will be able to select the most important features from a multidimensional dataset while explaining your decisions!

To use SHAP-Selection, you will need:
  * `SHAP <https://github.com/slundberg/shap>`_

-----------
Instalation
-----------

.. code:: python
 
       pip install shap-selection


-----------
Citation
-----------

.. code:: bibtex

       @INPROCEEDINGS{MarcilioJr2020shapselection,  
         author={W. E. {Marcílio} and D. M. {Eler}}, 
         booktitle={2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},   
         title={From explanations to feature selection: assessing SHAP values as feature selection mechanism},   
         year={2020},  
         pages={340-347},  
         doi={10.1109/SIBGRAPI51738.2020.00053}
       }


-----------
Usage 
-----------

To use SHAP-Selection, you must have a trained model. It works both for classification and regression purposes!

**Load a dataset**

.. code:: python

       iris_data = load_iris()

       X, y = iris_data.data, iris_data.target
       feature_names = np.array(iris_data.feature_names)

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


**Fit a model**

.. code:: python

       model = cb.CatBoostClassifier(verbose=False)    
       model.fit(X_train, y_train)

**Use SHAP-Selection**

.. code:: python

       from shap_selection import feature_selection

       # please, use agnostic = True to use with any model...
       # agnostic = False will only work with tree-based models
       feature_order = feature_selection.shap_select(model, X_train, X_test, feature_names, agnostic=False)


-----------
Support 
-----------

Please, if you have any questions feel free to contact me at wilson_jr@outlook.com
