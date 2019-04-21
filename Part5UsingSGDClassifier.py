#!/usr/bin/env python
# coding: utf-8

# In[28]:


# To follow Part 1 and Part 2 on these 3 main categories
# split by image path to the 3 main categories : Beauty, fashion, mobile
# categorise into Beauty Fashion and Mobile via information from image path
# make into modules for easy usage


# arg typeDataset only accept : 'beauty' , 'fashion', 'mobile'
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain
from nltk.collocations import *
import pickle
import os
import pickle
import numpy as np
from sklearn import metrics
import sys
def typeOfData():
    import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from itertools import chain
    import pickle
    import os
    import pickle
    import numpy as np
    from sklearn import metrics
    import sys
    
    typeDataset = input(" please type beauty , fashion  or mobile:")
    if typeDataset in ['beauty' , 'fashion' , 'mobile']:

        df = pd.read_csv('train.csv') # import training set

        image_path = df['image_path']
        df_beautyAll = df[image_path.str.contains('beauty_image', regex=True)]
        df_fashionAll = df[image_path.str.contains('fashion_image', regex=True)]
        df_mobileAll = df[image_path.str.contains('mobile_image', regex=True)]


        # In[29]:


        # manual change typeDataset ### attention!
        #typeDataset= 'beauty'
        #typeDataset = 'fashion'
        #typeDataset = 'mobile'


        # In[30]:


        # take 80% of each df_xxx and set as training while 20% of the training set is to test our accuracy of model
        lenDfBeauty = len(df_beautyAll)
        lenDfBeauty = round(0.001*(lenDfBeauty))  # take 80% as training set for classification
        df_beauty = df_beautyAll[0:lenDfBeauty]

        lenDfFashion = len(df_fashionAll)
        lenDfFashion = round(0.001*(lenDfFashion))  # take 80% as training set for classification
        df_fashion = df_fashionAll[0:lenDfFashion]

        lenDfMobile = len(df_mobileAll)
        lenDfMobile = round(0.001*(lenDfMobile))  # take 80% as training set for classification
        df_mobile = df_mobileAll[0:lenDfMobile]

        # from Part 6
        # take next 20% of each df_xxx and set as test set for trained model while first 80% was the training set for classification
        df_trainVerify_beauty = df_beautyAll[lenDfBeauty:round(1.02*lenDfBeauty)]
        df_trainVerify_fashion = df_fashionAll[lenDfFashion:round(1.02*lenDfFashion)]
        df_trainVerify_mobile = df_mobileAll[lenDfMobile:round(1.02*lenDfMobile)]

        print(df_trainVerify_mobile)
        print(len(df_trainVerify_mobile))
        print(df_trainVerify_mobile.iloc[len(df_trainVerify_mobile)-1])


        # In[31]:


        # tokenise to sentences # can skip since each cell is alr a sentence
        # for the 3 grps perform one by one, saving output as we go alon
        if typeDataset == 'beauty':
            df = df_beauty
        elif typeDataset == 'fashion':
            df = df_fashion
        elif typeDataset == 'mobile':
            df = df_mobile

        print(len(df_beauty))
        print(len(df_fashion))
        print(len(df_mobile))

        print(len(df['title']))


        sents = []
        for text in df['title']:
            

            sents.append(sent_tokenize(text))

        #print(len(sents))
            
        #print(len(sents))
        #print(type(sents[0]))
        #print(sents)

        #sentsUnlist = list(chain(*sents)) # unlist a list of lists   #################3 alert replace for all copies
        #print("sents unlisted :{}".format(sents))
        #print(sents)

        sentsUnlist = [' '.join(item) for item in sents]

        print(len(sents))
        print(len(sentsUnlist))

        #for item in sents:
        #    print(' '.join(item))
        #print('------------------------------')
        #for item in sentsUnlist:
        #    print(item)
              
        sents = sentsUnlist


        # In[32]:


        # Part 6 Tokenise to sentence
        # tokenise to sentences # can skip since each cell is alr a sentence
        


        if typeDataset == 'beauty':
            dftrainVerify = df_trainVerify_beauty 
        elif typeDataset == 'fashion':
            dftrainVerify = df_trainVerify_fashion 
        elif typeDataset == 'mobile':
            dftrainVerify = df_trainVerify_mobile


        sents1 = []
        for text in dftrainVerify['title']:
            sents1.append(sent_tokenize(text))
            


        sentsUnlist1 = [' '.join(item) for item in sents1]

        print(len(sents1))
        print(len(sentsUnlist1))

        #for item in sents:
        #    print(' '.join(item))
        #print('------------------------------')
        #for item in sentsUnlist:
        #    print(item)
              
        sents1 = sentsUnlist1


        # In[33]:


        # tokenise into words
        #print(sents)
        words = [word_tokenize(sent) for sent in sents]
        print(len(sents))
        print(len(words))
        print(len(df))
        #print(words)
        df = df.assign(tokenised=words)
        # unlist the list
        words = list(chain(*words)) # unlist a list of lists
        #print(words)
        #print(len(words))
        #print(words[0:10])
        #print(df[0:10])


        #print(df)


        # In[34]:


        # part 6 Tokenise into words
        #print(sents1)
        words1 = [word_tokenize(sent) for sent in sents1]
        #print(words)
        dftrainVerify = dftrainVerify.assign(tokenised=words1)
        # unlist the list
        #words1 = list(chain(*words1)) # unlist a list of lists
        #print(words)
        #print(len(words))
        #print(words[0:10])
        #print(df[0:10])


        #print(dftrainVerify)


        # In[35]:


        # use variable 'words'
        # checking for bigrams and sort according to frequency and filter out those freq more than 2
        # be sure to save as diff file name for the 3 categories, under pickle save
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)
        #finder = BigramCollocationFinder.from_words(list(df['removeStopwords']))
        bigrams = finder.ngram_fd.items()
        sorted(bigrams)



        # sort in descending order by frequency
        # Append to a list
        # https://stackoverflow.com/questions/28077573/python-appending-to-a-pickled-list 
        biscores = []
        for k,v in sorted(bigrams, key=lambda t:t[-1], reverse=True):
            #print(k,v)
            if v > 1:
                #print(k)
                biTog = k[0]+" "+k[1]
                #print((biTog))
                biscores.append(biTog)


        print(len(biscores))
        #print(biTog)


        # Now we "sync" our database
        #with open(bigram_filename,'wb') as wfp:
        #    pickle.dump(biscores, wfp)

        # Re-load our database
        #with open(bigram_filename,'rb') as rfp:
        #    biscores = pickle.load(rfp)
        #print(len(biscores))
        #print(biscores)

        # we need to group these words together so they make sense

        # SAVE as a data file
        # https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python/899199
        # to save as json, can see
        # https://stackoverflow.com/questions/890485/python-how-do-i-write-a-list-to-file-and-then-pull-it-back-into-memory-dict-re
        # to automate, we can change the file name we save as by determining the 'typeDataset' in cell above
        if typeDataset == 'beauty':
            saveNameBi = 'bigram_beauty_words.dat'
        elif typeDataset == 'fashion':
            saveNameBi = 'bigram_fashion_words.dat'
        elif typeDataset == 'mobile':
            saveNameBi = 'bigram_mobile_words.dat'

        # to write to a file called bigram_words.dat
        with open(saveNameBi, 'wb') as fp:
                pickle.dump(biscores, fp)
                #print(biscores)


        # In[36]:


        # use variable 'words'
        # checking for trigrams and filter out if freq of trigram is more than 2
        # be sure to save as diff file name for the 3 categories, under pickle save
        # run all 3 grps from the top to this cell, before proceeding to next cell
        
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(words)
        trigrams = finder.ngram_fd.items()
        sorted(trigrams)


        # sort in descending order by frequency
        # Append to a list
        # https://stackoverflow.com/questions/28077573/python-appending-to-a-pickled-list 


        triscores = []
        for k,v in sorted(trigrams, key=lambda t:t[-1], reverse=True):
            #print(k,v)
            if v > 1:
                #print(k)
                triTog = k[0]+" "+k[1]+" " +k[2]
                #print((triTog))
                triscores.append(triTog)


        print(len(triscores))
        #print(triTog)

        # to automate, we can change the file name we save as by determining the 'typeDataset' in cell above
        if typeDataset == 'beauty':
            saveNameTri = 'trigram_beauty_words.dat'
        elif typeDataset == 'fashion':
            saveNameTri = 'trigram_fashion_words.dat'
        elif typeDataset == 'mobile':
            saveNameTri = 'trigram_mobile_words.dat'

        with open(saveNameTri, 'wb') as fp:
                pickle.dump(triscores, fp)


        # In[37]:


        # using the function to remove bi and trigrams from df['title']
        import removeBiTri3Grp
        # run this one grp at a time, as removeBiTri3Grp output only runs through one set of biscores and triscores at a time
        lenDf = len(df)
        #print(biscores)

        # pre allocate list
        # https://stackoverflow.com/questions/311775/python-create-a-list-with-initial-capacity
        listMonogram = [None] * lenDf
        listBothBiTri = [None] * lenDf
        print(len(df))
        for i in range(0, lenDf):
            if i % 1000 == 0:
                print('i:{}'.format(i))
            #itemDescription = df['removeStopwords'][i]
            itemDescription = df['title'].iloc[i]
            #print("original: \n{}".format(itemDescription))
            
            #removebothBiTri, listTriBiThrown  = removeBiTri(itemDescription) # when itemDescription is already split into list
            removebothBiTri, listTriBiThrown = removeBiTri3Grp.removeBiTri3Grp(itemDescription.split(' '), typeDataset) # when itemDescription is a string
            #print(removebothBiTri, listTriBiThrown)
            listMonogram[i] = removebothBiTri
            listBothBiTri[i] = listTriBiThrown

        #print(df)
            


        # In[38]:


        df = df.assign(monogram = listMonogram)
        df = df.assign(bothBiTri = listBothBiTri)


        # In[39]:


        # part 6
        # now use the previous Bi and Trigram words created to check if these words are inside
        # run this one grp at a time, as removeBiTri3Grp output only runs through one set of biscores and triscores at a time
        #https://stackoverflow.com/questions/311775/python-create-a-list-with-initial-capacity
        import removeBiTri3Grp 

        lenDftrainVerify = len(dftrainVerify)
        #print(biscores)
        listMonogram1 = [None] * lenDftrainVerify
        listBothBiTri1 = [None] * lenDftrainVerify
        #print(dftrainVerify)
        print(lenDftrainVerify)
        for i in range(0, lenDftrainVerify):
            # print to see if running
            if i % 1000 == 0:
                print('i:{}'.format(i))
            #itemDescription = df['removeStopwords'][i]
            itemDescription = dftrainVerify['title'].iloc[i]
            #print("original: \n{}".format(itemDescription))
            
            #removebothBiTri, listTriBiThrown  = removeBiTri(itemDescription) # when itemDescription is already split into list
            removebothBiTri, listTriBiThrown = removeBiTri3Grp.removeBiTri3Grp(itemDescription.split(' '), typeDataset) # when itemDescription is a string
            #print(removebothBiTri, listTriBiThrown)
            listMonogram1[i] = removebothBiTri
            listBothBiTri1[i] = listTriBiThrown
        #print(dftrainVerify)
            


        # In[ ]:





        # In[40]:


        # Part 6
        # idea is to create a list with initial value then add to it
        # faster than append to the whole df
        dftrainVerify = dftrainVerify.assign(monogram = listMonogram1)
        dftrainVerify = dftrainVerify.assign(bothBiTri = listBothBiTri1)


            
        # In[41]:


        # remove stop words after removed bi and trigrams
        from nltk.corpus import stopwords 
        from string import punctuation
        from itertools import chain
        import pickle
        from stop_words import get_stop_words
        # the 2nd stop_words library contains indonesian, the first did not
        print('here')
        customStopWords=set(stopwords.words('english')+list(punctuation) + get_stop_words('indonesian')+ get_stop_words('english'))
        print('here')
        # remove stop words

        wordsWOStopwords = []
        for text in df['monogram']:
            text = ' '.join(text)
            #print(text)
            wordsWOStopwords.append([word for word in word_tokenize(text.lower()) if word not in customStopWords])
        print('here')
        df = df.assign(removeStopwords=wordsWOStopwords)
        #del df['tokenised']
        #print(df)
        # unlist the list
        wordsWOStopwords = list(chain(*wordsWOStopwords)) # unlist a list of lists
        #print("wordsWOStopwords unlisted :{}".format(wordsWOStopwords))
        print('here')

        # export as a file of words without stop words, this is to save a variable
        #with open('noStopwords_words.dat', 'wb') as fp:
        #    pickle.dump(wordsWOStopwords, fp)


        # In[42]:


        # remove stop words after removed bi and trigrams
        from nltk.corpus import stopwords 
        from string import punctuation
        from itertools import chain
        import pickle
        from nltk.tokenize import word_tokenize, sent_tokenize
        from stop_words import get_stop_words
        # the 2nd stop_words library contains indonesian, the first did not

        customStopWords=set(stopwords.words('english')+list(punctuation)+ get_stop_words('indonesian'))

        # remove stop words

        wordsWOStopwords = []
        for text in dftrainVerify['monogram']:
            text = ' '.join(text)
            #print(text)
            wordsWOStopwords.append([word for word in word_tokenize(text.lower()) if word not in customStopWords])

        dftrainVerify = dftrainVerify.assign(removeStopwords=wordsWOStopwords)
        #del df['tokenised']
        #print(dftrainVerify)
        # unlist the list
        wordsWOStopwords = list(chain(*wordsWOStopwords)) # unlist a list of lists
        #print("wordsWOStopwords unlisted :{}".format(wordsWOStopwords))


        # export as a file of words without stop words, this is to save a variable
        #with open('noStopwords_words.dat', 'wb') as fp:
        #    pickle.dump(wordsWOStopwords, fp)


        # In[43]:


        # write df to pickle
        # to automate, we can change the file name we save as by determining the 'typeDataset' in cell above
        import pickle
        if typeDataset == 'beauty':
            fnremoveBiTriStop = 'df_removeTriBiStopBeauty.dat'
        elif typeDataset == 'fashion':
            fnremoveBiTriStop = 'df_removeTriBiStopFashion.dat'
        elif typeDataset == 'mobile':
            fnremoveBiTriStop = 'df_removeTriBiStopMobile.dat'
            
        with open(fnremoveBiTriStop, 'wb') as fp:
            pickle.dump(df, fp)
        print(df)

        # part 1 ends here


        # In[44]:


        # open back the file just saved
        import pickle
        if typeDataset == 'beauty':
            fnremoveBiTriStop = 'df_removeTriBiStopBeauty.dat'
        elif typeDataset == 'fashion':
            fnremoveBiTriStop = 'df_removeTriBiStopFashion.dat'
        elif typeDataset == 'mobile':
            fnremoveBiTriStop = 'df_removeTriBiStopMobile.dat'
            
        with open(fnremoveBiTriStop, 'rb') as fp:
            df = pickle.load(fp)


        # In[45]:

        # stemming the column "removeStopwwords" which is the final column after removing Stopwords and Bi Trigrams
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.stem.lancaster import LancasterStemmer
        st=LancasterStemmer()


        stemmingWord = []
        for sentence in df['removeStopwords']:
            #print()
            #print(sentence)
            sentence = ' '.join(sentence)
            #print(sentence)
            stemmedWords=[st.stem(word) for word in word_tokenize(sentence)]
            #print(stemmedWords)
            stemmingWord.append([stem for stem in stemmedWords])
        df = df.assign(stemWords=stemmingWord)
        print(df)

        # In[46]:


      
        # Part 6
        # stemming the column "removeStopwwords" which is the final column after removing Stopwords and Bi Trigrams
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.stem.lancaster import LancasterStemmer
        st=LancasterStemmer()


        stemmingWord1 = []
        for sentence1 in dftrainVerify['removeStopwords']:
            #print()
            #print(sentence1)
            sentence1 = ' '.join(sentence1)
            #print(sentence1)
            stemmedWords1=[st.stem(word) for word in word_tokenize(sentence1)]
            #print(stemmedWords)
            stemmingWord1.append([stem for stem in stemmedWords1])
        dftrainVerify = dftrainVerify.assign(stemWords=stemmingWord1)
        print(dftrainVerify)
        
        # In[47]:


        # collate all words and put in 'mergeAll' column
        import pickle

        # make words from bigram and trigram to one word each
        # using the function to remove bi and trigrams from df['title']
        lenDf = len(df)
        listMergeAll = []
        for i in range(0, lenDf):
            thisRowMono = df['stemWords'].iloc[i]
            thisRowBiTri = df['bothBiTri'].iloc[i]
            RowMonoStr = ' '.join(thisRowMono)
            BitriNoSpace = ' '.join(thisRowBiTri)
            mergestr = "{} {}".format(BitriNoSpace,RowMonoStr)
            listMergeAll.append(mergestr)

        df = df.assign(mergeAll = listMergeAll)
                

        #print(df)
             
        # write df to pickle
        # to automate, we can change the file name we save as by determining the 'typeDataset' in cell above

        if typeDataset == 'beauty':
            fnMergeAll = 'df_beauty_mergeAll.dat'
        elif typeDataset == 'fashion':
            fnMergeAll = 'df_fashion_mergeAll.dat'
        elif typeDataset == 'mobile':
            fnMergeAll = 'df_mobile_mergeAll.dat'
            
        # writing
        with open(fnMergeAll, 'wb') as fp:
            pickle.dump(df, fp)

            


        # In[48]:


        # Part 6
        # collate all words and put in 'mergeAll' column

        # make words from bigram and trigram to one word each
        # using the function to remove bi and trigrams from df['title']
        lenDftrainVerify = len(dftrainVerify)
        listMergeAll = []
        for i in range(0, lenDftrainVerify):
            thisRowMono = dftrainVerify['stemWords'].iloc[i]
            thisRowBiTri = dftrainVerify['bothBiTri'].iloc[i]
            RowMonoStr = ' '.join(thisRowMono)
            BitriNoSpace = ' '.join(thisRowBiTri)
            mergestr = "{} {}".format(BitriNoSpace,RowMonoStr)
            listMergeAll.append(mergestr)

        dftrainVerify = dftrainVerify.assign(mergeAll = listMergeAll)
                
                

        #print(dftrainVerify)
            
        # pickle save 
        # write df to pickle

        if typeDataset == 'beauty':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_beauty.dat'
        elif typeDataset == 'fashion':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_fashion.dat'
        elif typeDataset == 'mobile':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_mobile.dat'


        with open(fnMergeAllVerify, 'wb') as fp:
            pickle.dump(dftrainVerify, fp)


        # In[49]:


        # load back the pickle to check
        with open(fnMergeAll, 'rb') as fp:
            df = pickle.load(fp)
        #print(df)

        # since there are 3 groups, we should use a automated process to determine open which file
        if typeDataset == 'beauty':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_beauty.dat'
        elif typeDataset == 'fashion':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_fashion.dat'
        elif typeDataset == 'mobile':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_mobile.dat'

            

        # Load the pickle of the test training set in Part 6
        with open(fnMergeAllVerify, 'rb') as fp:
            dftrainVerify = pickle.load(fp)
        print(dftrainVerify)


        # In[50]:


        # Alternative,  testing with another classifier model 
        # result showed this model has higher accuracy rate, so we go with this classifier
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import SGDClassifier
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf',TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                               alpha=1e-4, random_state=42,
                                  max_iter=5, tol=None)),
        ])


        # In[51]:


        # fitting into the pipeline to train
        text_clf.fit(df['mergeAll'], df['Category']) 


        # In[52]:

        if typeDataset == 'beauty':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_beauty.dat'
        elif typeDataset == 'fashion':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_fashion.dat'
        elif typeDataset == 'mobile':
            fnMergeAllVerify = 'dftrainVerify_mergeAll_TrainVerify_mobile.dat'


        with open(fnMergeAllVerify, 'rb') as fp:
            dftrainVerify = pickle.load(fp)


        # In[53]:


        # using the pipeline on a training data to predict
        predicted =  text_clf.predict(dftrainVerify['mergeAll'])

        dftrainVerify = dftrainVerify.assign(predicted=predicted)

        if typeDataset == 'beauty':
            fntext= 'beautyTraining.csv'
        elif typeDataset == 'fashion':
            fntext= 'fashionTraining.csv'
        elif typeDataset == 'mobile':
            fntext= 'mobileTraining.csv'
        dftrainVerify.to_csv(fntext)


        # In[54]:


        # Evaluation on performance of test
        # this part if you use "test", need to upload to kaggle to see result
        # predicted is done above
        print(np.mean(predicted ==  dftrainVerify.Category))


        print(metrics.classification_report(dftrainVerify.Category, predicted))


        # In[ ]:





        # In[55]:


        # tuning on traning set
        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

        from sklearn.model_selection import GridSearchCV
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2),(1,3),(1,4)],
            #'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-4, 1e-3),
        }
        gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

        #gs_clf = gs_clf.fit(df['mergeAll'][:400], df['Category'][:400])
        gs_clf = gs_clf.fit(df['mergeAll'], df['Category'])
        #print(df['mergeAll'][:400])


        # In[56]:


        # tuning on training set
        #a = gs_clf.predict(['4gb32gbblack'])
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


    else:
        sys.exit("Please give a valid input")


if __name__ == "__main__":
    typeOfData()
