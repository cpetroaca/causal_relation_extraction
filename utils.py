import numpy as np

class Vectorizer:
    def __init__(self, word2Idx, labelsMapping, minDistance, maxDistance, maxSentenceLen=100):
        self.word2Idx = word2Idx
        self.labelsMapping = labelsMapping
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.maxSentenceLen = maxSentenceLen
        self.distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
        for dis in range(self.minDistance,self.maxDistance+1):
            self.distanceMapping[dis] = len(self.distanceMapping)
    
    def vectorizeInput(self, file):
        """Creates matrices for the events and sentence for the given file"""
        labels = []
        positionMatrix1 = []
        positionMatrix2 = []
        tokenMatrix = []
        
        for line in open(file):
            splits = line.strip().split('\t')
            
            label = splits[0]
            pos1 = splits[1]
            pos2 = splits[2]
            sentence = splits[3]
            tokens = sentence.split(" ")
            
            tokenIds = np.zeros(self.maxSentenceLen)
            positionValues1 = np.zeros(self.maxSentenceLen)
            positionValues2 = np.zeros(self.maxSentenceLen)
            
            for idx in range(0, min(self.maxSentenceLen, len(tokens))):
                tokenIds[idx] = self.getWordIdx(tokens[idx])
                
                distance1 = idx - int(pos1)
                distance2 = idx - int(pos2)
                
                if distance1 in self.distanceMapping:
                    positionValues1[idx] = self.distanceMapping[distance1]
                elif distance1 <= self.minDistance:
                    positionValues1[idx] = self.distanceMapping['LowerMin']
                else:
                    positionValues1[idx] = self.distanceMapping['GreaterMax']
                    
                if distance2 in self.distanceMapping:
                    positionValues2[idx] = self.distanceMapping[distance2]
                elif distance2 <= self.minDistance:
                    positionValues2[idx] = self.distanceMapping['LowerMin']
                else:
                    positionValues2[idx] = self.distanceMapping['GreaterMax']
                
            tokenMatrix.append(tokenIds)
            positionMatrix1.append(positionValues1)
            positionMatrix2.append(positionValues2)
            
            labels.append(self.labelsMapping[label])
            
        return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
    def getWordIdx(self, token): 
        """Returns from the word2Idex table the word index for a given token"""       
        if token in self.word2Idx:
            return self.word2Idx[token]
        elif token.lower() in self.word2Idx:
            return self.word2Idx[token.lower()]
        
        return self.word2Idx["UNKNOWN_TOKEN"]