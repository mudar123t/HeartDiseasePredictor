# Working with the old Dataset
1) run the project on the old (heart_disease) dataset inside "notebooks" foleder,
2) we did the exploration inside "01_exploration.ipynb", then the preprocessing inside "02_preprocessing.ipynb",
3) we did some feature engineering inside "03_feature_selection.ipynb"
  with the help of "feature_selection.py" file inside src folder
4) we run the modeling inside "04_modeling.ipynb" file, and got bad training output,
5) we noticed that there could be leak because the feature selection and the training is happening in seperated files
  so we moved our process into a single file named "04_feature_modeling.ipynb" where we did everyting in a single place
6) we kept getting bad resutls which was suspicios, so we ran some codes to check if the problem was from the dataset
  and indeed it was from the dataset, so we found another dataset from kaggle (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
  to check it, we ran the previouse ran codes to check the dataset and we got a better resutls so we moved into the next part of the project
# Working with the new Dataset
1) we did all the process and the project for the new dataset inside "notebooks_new" folder
2) inside that folder we ran a similar code that was inside "01_exploration" to check the data and there was 0 null values
  so we moved to the "02_feature_modeling" to apply the feature selection and the training,
3) since the new dataset had fewer columns, we took all of them, but still we ran the feature methods in order to show the most important ones
4) we run the old models (without chaing their variabels) on the new dataset, and we got a whole different results
  instead of getting bad results (values between the 0.2 and 0.6) now we noticed that there is overfitting that is happeninig
5) we started optimising the models.py files variabels to get the best outcome without overfitting
6) after so many tries we got the a data that is near desired one, with only 2 models that are near to overfitting
7) we saved the resutls and started runing the evaluations and showing the data inside "03_evaluation" file with plot to represent the data we got