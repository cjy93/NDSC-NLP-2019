# testing the algo for removing dual and trigrams from a sentence
# make it a function
import pickle

try:
    with open ('bigram_beauty_words.dat', 'rb') as fp:
            biscores_beauty = pickle.load(fp)
except FileNotFoundError:
    pass

try:
    with open ('trigram_beauty_words.dat', 'rb') as fp:
            triscores_beauty = pickle.load(fp)
except FileNotFoundError:
    pass
        
try:        
    with open ('bigram_fashion_words.dat', 'rb') as fp:
            biscores_fashion = pickle.load(fp)
except FileNotFoundError:
    pass
        
try:
    with open ('trigram_fashion_words.dat', 'rb') as fp:
            triscores_fashion = pickle.load(fp)
except FileNotFoundError:
    pass

try:        
    with open ('bigram_mobile_words.dat', 'rb') as fp:
            biscores_mobile = pickle.load(fp)
except FileNotFoundError:
    pass

try:
    with open ('trigram_mobile_words.dat', 'rb') as fp:
            triscores_mobile = pickle.load(fp)
except FileNotFoundError:
    pass
    

def removeBiTri3Grp(splits , typeDataset):
    # to automate, we can change the file name we save as by determining the 'typeDataset' in cell above
    if typeDataset == 'beauty':
        biscores = biscores_beauty
        triscores = triscores_beauty
    elif typeDataset == 'fashion':
        biscores = biscores_fashion
        triscores = triscores_fashion
    elif typeDataset == 'mobile':
        biscores = biscores_mobile
        triscores = triscores_mobile

        
    ################################################################
    
    #testStr = 'mon dual pair tri an gle mona'
    #dualList = ['dual pair']
    #triList = ['mon tri an']

    #splits = testStr.split(' ')
    #splits =['snail', 'white', 'cream', 'original', '100']
    #print(splits)

    # remove trigram
    #for i in range(len(splits)-1 , 1, -1):
    
    listTriBiThrown = []
    i = len(splits)-1     
    while i > 1:
        #print(i)
        #print('starting with list of')
        #print(splits)
        #print(splits[i])
        #print(splits[i-1])
        #print(splits[i-2])
        tri = "%s %s %s" % (splits[i-2], splits[i-1],splits[i])
        #print("i:%i, dual: %s" % (i, dual))
        #print(tri)
        
        if tri in triscores:
            # pop i, i-1 , i-2
            #print(tri)
            splits.pop(i)
            splits.pop(i-1)
            splits.pop(i-2)
            triNoSpace = ''.join(tri.split(' ')) # remove space so later when i merge all words per sentence, these will come tog
            listTriBiThrown.append(triNoSpace)
            i = i-2 # just incase remove 2 other items        
        i = i-1  # to prevent infinite looping
        #print('i: %i, %i' % (i,i+1))

    # remove bigram
    #for i in range(len(splits)-1, 0, -1):
    i = len(splits)-1
    while i > 0:
        #print("i:{}, i-1:{}".format(i, i-1))
        #print(splits[i])
        #print(splits[i-1])
        dual = "%s %s" % (splits[i-1], splits[i])  # remove from the back so index not changed
        #print("i:%i, dual: %s" % (i, dual))
        #print(dual)
        if dual in biscores:
            # pop i, i-1
            #print(dual)
            splits.pop(i)
            splits.pop(i-1)
            dualNoSpace = ''.join(dual.split(' ')) # remove space so later when i merge all words per sentence, these will come tog
            listTriBiThrown.append(dualNoSpace)
            #print('after pop')
            #print(splits)
            i = i-1 # since the i-1 position may also be eliminated
        i = i-1   # to go to next round / next pair
        #print('i: %i, %i' % (i,i+1))
    #print(splits)

    #print('#######################################################')

    #print(splits)
    return(splits, listTriBiThrown)