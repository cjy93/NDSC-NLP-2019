# Classification techniques and NLP for NDSC FEB 2019
## Project for NLP Feb 2019  
Special thing about my method that MIGHT BE most different other than those basic cleaning is I first partitioned the dataset into the 3 subcategories "Beauty", "fashion" and "mobile" before i do the NLP on each category separately. Do this step for both the train set and test set.  

Additionally, we also built our own Bigram and Trigram identifier because if we use NLTK version, they only make our bigram and trigram AFTER we remove stopwords, which may lead to inaccuracy. We built our Bigram and Trigram before remove stopwords to make it more accurate, just incase the stopwords are part of a name.

For each of the set, i eventually choose SGD classifier on each subgroup to get the predicted column.
