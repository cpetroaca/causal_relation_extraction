"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""
from __future__ import print_function
import numpy as np
import gzip
import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
from utils import *

folder = 'data/'
outputFilePath = folder + 'causal-relations.pkl.gz'
#We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
embeddingsPath = folder + 'wiki_extvec.gz'
files = [folder+'ctrain.txt', folder+'ctest.txt']

#Mapping of the labels to integers
labelsMapping = {'Other':0, 'Cause-Effect(e1,e2)':1, 'Cause-Effect(e2,e1)':2}
words = {}
maxSentenceLen = [0,0]
minDistance = -30
maxDistance = 30

for fileIdx in range(len(files)):
    file = files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]

        sentence = splits[3]        
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True
            

print("Max Sentence Lengths: ", maxSentenceLen)
        
# :: Read in word embeddings ::
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

# :: Downloads the embeddings from the York webserver ::
if not os.path.isfile(embeddingsPath):
    basename = os.path.basename(embeddingsPath)
    if basename == 'wiki_extvec.gz':
           print("Start downloading word embeddings for English using wget ...")
           #os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
           os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_english_embeddings/"+basename+" -P embeddings/")
    else:
        print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
        exit()
        
# :: Load the pre-trained embeddings file ::
fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
	
print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode('utf-8').strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)
       
        
wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))

# :: Create token matrix ::
vectorizer = Vectorizer(word2Idx, labelsMapping, minDistance, maxDistance, max(maxSentenceLen))
train_set = vectorizer.vectorizeInput(files[0])
test_set = vectorizer.vectorizeInput(files[1])

data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 
        'train_set': train_set, 'test_set': test_set, 'labels_mapping': labelsMapping, 'max_sentence_length': max(maxSentenceLen), 'min_distance': minDistance, 'max_distance': maxDistance}

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder") 