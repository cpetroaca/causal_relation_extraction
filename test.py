import gzip
import numpy as np
from keras.models import load_model
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
from utils import *

folder = 'data/'
files = [folder+'my_ctest.txt']

print("Load dataset")
f = gzip.open(folder + 'causal-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

maxSentenceLen = data['max_sentence_length']
word2Idx = data['word2Idx']
labelsMapping = data['labels_mapping']
minDistance = data['min_distance']
maxDistance = data['max_distance']
            
print("Max Sentence Lengths: ", maxSentenceLen)

vectorizer = Vectorizer(word2Idx, labelsMapping, minDistance, maxDistance, maxSentenceLen)
yTest, sentenceTest, positionTest1, positionTest2 = vectorizer.vectorizeInput(files[0])

model = load_model('model/causal_rel_model.h5')

pred_test_ini = model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
pred_test = pred_test_ini.argmax(axis=-1)
print("test result:")
print(pred_test)