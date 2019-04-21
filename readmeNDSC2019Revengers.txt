*************** READ ME*************************
How to use the files?
************************************************
Overall Classification model came from this website : https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

** To test the accuracy of our model, we use Part5(train model and use to test on part6 data) and Part6(20% of the dataset). I have split 80% of the train data to train the model and 20% of the train data to check how accurately out training model works. This cannot be done on test.csv since there is no 'Category' column to check against for the accuracy of result** 

1) Run part5 3 times, by selecting 'beauty' ,'fashion', or 'mobile' manually on cell 2. This can be done by removing and adding '#' infront of the desired 'typeDataset= xxx'

2) Cell 2 should only have one selection of 'typeDataset' at any one time


###################################

** To test the accuracy of our model, run part7 (run the model on test set). **

1) Run part7 3 times, by selecting 'beauty' ,'fashion', or 'mobile' manually on cell 2. This can be done by removing and adding '#' infront of the desired 'typeDataset= xxx'

2) Cell 2 should only have one selection of 'typeDataset' at any one time

For tuning of which model to use, please refer to scikit learn website to find better models if you think theres something better than 'SGDClassifier'. 
Refer to: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

3) notice unlike part5, part7 does not have a predictor value in percentages since test.csv does not have column on 'Category', so nothing to compare the accuracy of prediction

4) Notice that there are some cells at the bottom for " # tuning', that is tuning for the Pipeline model parameters.

5) Finally, run Part8 to merge the 3 output csv from part7 and make them into the submission format to upload to Kaggle
