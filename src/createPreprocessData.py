from __future__ import division
from __future__ import print_function
import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import pandas as pd
import nltk
import sklearn
import os

class FilePaths:
        import sklearn
        "filenames and paths to data"
        fnCharList = '../model/charList.txt'
        fnAccuracy = '../model/accuracy.txt'
        fnTrain = '../data/'
        fnInfer ='../data/test.png'
        fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0]);
    return recognized[0]

def inferBatch(model):
    "recognize text in set of images provided by file path"
    
    col_names =  ['Jacard Similarity', 'Cosine similarity', 'Levenstein Similarity', 'Euclidian distance', 'Character Level Accuracy', 'Actual image' , 'Predicted Text']
    df  = pd.DataFrame(columns = col_names)
    f=open(FilePaths.fnTrain+'validation.txt')
        
    for line in f:
            # ignore comment line
            if not line or line[0]=='#':
                continue
            
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            
            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = FilePaths.fnTrain + 'validation/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            # GT text are columns starting at 9
            gtText = DataLoader.truncateLabel('self', ' '.join(lineSplit[8:]), Model.maxTextLen)
            

            # check if image is not empty
            if not os.path.getsize(fileName):
                #bad_samples.append(lineSplit[0] + '.png')
                continue

            # put sample into list
            #self.samples.append(Sample(gtText, fileName))
            img = preprocess(cv2.imread(fileName, cv2.IMREAD_GRAYSCALE), Model.imgSize)
            batch = Batch(None, [img])
            try:
                (recognized, probability) = model.inferBatch(batch, True)
                print('Recognized:', '"' + recognized[0] + '"')
                print('Probability:', probability[0]);
                jaccard_score,cosine_score,levens_score,euclidean_score = get_AccuracyMetrics(gtText, recognized[0])
                acc = getCharacterLevelAccuracy(gtText,recognized[0])
                print('Character Accuracy:', acc)
                df.loc[len(df)] = [jaccard_score, cosine_score, levens_score, euclidean_score, acc, gtText, recognized[0]]
            except:
                continue
        
    df.to_csv(r'C:\Projects\Handwriting recognition\Validation_IAM_HAN.csv')    
    
def getCharacterLevelAccuracy(a,b):
    length = len(a)
    correctCount = 0
    for x, y in zip(a, b):
        if x == y:
            correctCount = correctCount+1
    if correctCount == 0:
        return '0%'
    else:
        per = (correctCount/length)*100
        return str(per)+'%'
        
def get_AccuracyMetrics(actual_text,detected_text):
    news_headline1 = a = actual_text
    news_headline2 = b = detected_text
    news_headlines = [news_headline1, news_headline2]
    news_headline1_tokens = nltk.word_tokenize(news_headline1)
    news_headline2_tokens = nltk.word_tokenize(news_headline2)
    try:           
        transformed_results = transform([news_headline1_tokens, news_headline2_tokens])
    except:
        return None,None,None,None
    
            
    print('Euclidian Distance (lower the distance, more is the acuracy\n')
    print('======================')
            
    print('Master Sentence: %s' % news_headlines[0])
    for i, news_headline in enumerate(news_headlines):
                euclidean_score = sklearn.metrics.pairwise.euclidean_distances([transformed_results[i]], [transformed_results[0]])[0][0]
                print('-----')
                print('Score: %.2f, Comparing Sentence: %s' % (euclidean_score, news_headline))
                
    print('\nCosine Similarity\n')
    print('======================')
            
    print('Master Sentence: %s' % news_headlines[0])
    for i, news_headline in enumerate(news_headlines):
                cosine_score = sklearn.metrics.pairwise.cosine_similarity([transformed_results[i]], [transformed_results[0]])[0][0]
                print('-----')
                print('Score: %.2f, Comparing Sentence: %s' % (cosine_score, news_headline))
            
    print('\nJaccard Similarity\n')
    print('======================')
            
    print('Master Sentence: %s' % a)
    score = get_jaccard_sim(a,a)
    print('-----')
    print('Score: %.2f, Comparing Sentence: %s' % (score, a))
    jaccard_score = get_jaccard_sim(a,b)
    print('-----')
    print('Score: %.2f, Comparing Sentence: %s' % (jaccard_score, b))

    print('\nLevenshtein Similarity\n')
    print('======================')
 
    print('Master Sentence: %s' % a)
    score = levens_similarity(a,a)
    print('-----')
    print('Score: %.2f, Comparing Sentence: %s' % (score, a))
    levens_score = levens_similarity(a,b)
    print('-----')
    print('Score: %.2f, Comparing Sentence: %s' % (levens_score, b))
    
    return jaccard_score,cosine_score,levens_score,euclidean_score

    
def get_jaccard_sim(str1, str2):
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def cosine_similarity(actual_text,detected_text):
    import sklearn 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_detected = tfidf_vectorizer.fit_transform([detected_text])
    tfidf_matrix_actual = tfidf_vectorizer.fit_transform([actual_text])
    return cosine_similarity(tfidf_matrix_actual, tfidf_matrix_detected)

def euclidian_measure(actual_text,detected_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import euclidean_distances
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_detected = tfidf_vectorizer.fit_transform([detected_text])
    tfidf_matrix_actual = tfidf_vectorizer.fit_transform([actual_text])
    return euclidean_distances(tfidf_matrix_actual, tfidf_matrix_detected)

    
def levens_similarity(actual_text,detected_text):
    import Levenshtein
    return Levenshtein.ratio(actual_text,detected_text)




def transform(headlines):
    tokens = [w for s in headlines for w in s ]
    #print()
    #print('All Tokens:')
    #print(tokens)
    import sklearn
    results = []
    label_enc = sklearn.preprocessing.LabelEncoder()
    onehot_enc = sklearn.preprocessing.OneHotEncoder()
    
    encoded_all_tokens = label_enc.fit_transform(list(set(tokens)))
    encoded_all_tokens = encoded_all_tokens.reshape(len(encoded_all_tokens), 1)
    
    onehot_enc.fit(encoded_all_tokens)
    
    for headline_tokens in headlines:
        #print()
        #print('Original Input:', headline_tokens)
        
        encoded_words = label_enc.transform(headline_tokens)
        #print('Encoded by Label Encoder:', encoded_words)
        
        encoded_words = onehot_enc.transform(encoded_words.reshape(len(encoded_words), 1))
        #print('Encoded by OneHot Encoder:')
        #print(encoded_words)
        import numpy as np
        results.append(np.sum(encoded_words.toarray(), axis=0))
    
    return results

def main():
    import nltk
    import sklearn
    from numpy import argmax
    import numpy as np
    # optional command line args
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', help='Path to test image', action='store_true')
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    parser.add_argument('--infer', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset    
    if args.train or args.validate or args.infer:
    # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

    # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
        
    # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)
        elif args.infer:
            model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
            inferBatch(model)
            

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        detected_text = infer(model, FilePaths.fnInfer)
    
if __name__ == '__main__':
    main()

