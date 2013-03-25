'''
Script to train NB classifier on POS tagged sents
based on a shortlist of common verbs identified to have
spatial aspect.

Currently removing sentences with quotes and question sentences.

@tturp '13
'''


import nltk
from nltk.stem import PorterStemmer
import itertools
import scipy


#brown = nltk.corpus.brown.tagged_words()
brown = nltk.corpus.brown.tagged_sents(categories=['science_fiction','adventure','mystery','romance','science_fiction'])
#brown = nltk.corpus.brown.tagged_sents(categories=['religion','news','humor','lore','editorial','fiction','government'])
#OUT_FILE = "./NikhilThesisData.txt"
OUT_FILE= "./5Shorts005.txt"

verbtags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
shortList = ['enter',
             'exit',
             'leav','left',#stemmed leave
             'pass',
             'fall',
             'sit','sat',
             'lay',
             'rest',
             'perch',
             'stand','stood',
             'cross',
             'forg',#stemmed forge
             'jump',
             'pierce',
             'put',
             'set',
             'insert',
             'run','ran',
             'walk',
             'fli',#stemmed fly
             'balanc',#stemmedbalance
             'crawl',
             'roll',
             'gallup'
             ]

#List verbs we don't want, ever
outMains = ['has','had','have','having'
            'do','did','done',
            'was','is','be',
            'should','could','would',"won't"
            ]
punc = ['.','?','!']
             
sents = []; sent = []; verbs = {};
curVerbs = []#verbs in the sentence
verbsent = False
ps = PorterStemmer()
MIN_VERB_LENGTH = 4
MAX_SENT_LENGTH = 20

def handleVerbs(curVerbs, verbs, sent):
    '''
        @input curVerbs: verbs in the sentence
        @input verbs: verb dictionary
        @input sent: sentence to be stored in verb dictionary

    '''
    #Take a list of curVerbs and the dictionary verbs{},
    #append sent to that verb's dictionary entry if it exists
    #else create the entry
    stemmed = ""
    
    if not(tooModal(curVerbs)):
        for verb in curVerbs:
            stemmed = ps.stem(verb)
            if stemmed in verbs:
                #If in the verb dictionary already, append
                verbs[stemmed].append(sent)
            else:
                #else create a new dictionary entry
                verbs[stemmed] = [sent]

def tooModal(verbs):
    #If more than half the verbs are modals, return false
    countBadModals = 0
    for word in verbs:
        if word in outMains:
            countBadModals += 1
    return (countBadModals >= (len(verbs)/2))

def train(verbDict):
    '''
        For verbs in the shortList, build featureset
        For verbs not in the shortList, build featureset
        Train classifier based on features
        
    '''
    shorts = []#Verb sentences with verbs in the shortlist
    nots = []#Verb sentences without verbs in the shortlist

    #Extract unique sentences with verbs from the shortlist
    for verb in verbDict:
        if verb in shortList:
            for sent in verbDict[verb]:
                if not sent in shorts:
                    sent = preprocess_positive_observation(sent,verb)
                    shorts.append(sent)#extracted = extract_verbFeatures(verbDict[verb])
        else:
            for sent in verbDict[verb]:
                if not sent in nots:
                    sent = preprocess_negative_observation(sent,verb)
                    nots.append(sent)#extracted = extract_verbFeatures(verbDict[verb])
    notFeatures = extract_posSent_features(nots,False)
    shortFeatures = extract_posSent_features(shorts,True)
##    return (nots,shorts)  
    #return nltk.MaxentClassifier.train([(notFeatures,'False'),(shortFeatures,'True')],'GIS')
    return nltk.NaiveBayesClassifier.train([(notFeatures,'False'),(shortFeatures,'True')])

def preprocess_positive_observation(sent,verb):
    '''
        Feature preprocessing
        The classifier tends to only pick the sentences with our short verbs
        So remove the short verbs from positive examples
    '''
    #Do nothing right now
    return sent

def preprocess_negative_observation(sent,verb):
    '''
        Do nothing right now
    '''
    return sent
        
def extract_posSent_features(sents, pullShortlist):
    '''
        Given list of POS tagged sentences, extract features for each word

        @input sents: list of POS tagged sentences

        @return dict: dictionary of tuples with each word in sentences and True

    '''
    #bag-of-words
    dic = dict([('contains-word(%s)' % w[0], True) for w in list(itertools.chain(*sents))])
    if pullShortlist:
        for item in shortList:
            if 'contains-word(%s)'%item in dic:
                dic.pop('contains-word(%s)'%item)
    return dic

def getModel():
    return model

def postproc_sents(sent):
    dpunc = ['!',';','.']
    obsolete = ['--']
    remove = ['?',':',')']
    endquote = "''"
    startquote = '``'
    words = [w[0] for w in sent]
    pos = [w[1] for w in sent]
    sentVerbs = []
    for word in sent:
        if word[1] in verbtags:
            sentVerbs += word[0]

    if tooModal(sentVerbs):
        #Too many modals, reject
        return ""
    
    if sent[-1][0] in remove:
        #No question sentences
        return ""
    if sent[-2][0] in dpunc:
        #Remove Brown double punctuation
        sent.remove(sent[-1])
    if sent[-1][0] in obsolete:
        #Remove obsolete punctuation
        sent.remove(sent[-1])
    if startquote in words and not endquote in words:
        #If startquote and no endquote, and endquote
        if sent[-1][0] == '.':
            sent.remove(sent[-1])
            sent.append((endquote,endquote))
            sent.append(('.','.'))
        else:
            #Append period if needed
            sent.append((endquote,endquote))
            sent.append(('.','.'))
    if endquote in words and not startquote in words:
        #If endquote and no startquote
        sent = [(startquote,startquote)] + sent
    if sent[0][0] in remove:
        sent.remove(sent[0])
    if not sent[-1][0] in dpunc:
        sent.append(('.','.'))


#post-postproc:
    words = [w[0] for w in sent]
    if startquote in words:
        #Remove sents with quotes
        return ""
    if sent[0][0][0].islower():
        #Capitalize the first letter of the sentence
        item = sent[0]
        word = item[0]
        pos = item[1]
        word = word[0].capitalize()+word[1:]
        sent[0] = (word,pos)
    
    return sent

'''
Iterate through brown corpus of tagged english sentences.
Load every tagged verb's sentence into the dictionary verbs{}
'''
for sent in brown:
    for item in sent:
        if item[1] in verbtags:
            verbsent = True
            curVerbs.append(item[0])
    if MAX_SENT_LENGTH > len(sent) > MIN_VERB_LENGTH :
        if(verbsent):
            sents.append(sent)
            handleVerbs(curVerbs,verbs,sent)
            verbsent = False
    curVerbs = []

trainSentCount = 0
for item in verbs:
    for entry in verbs[item]:
        trainSentCount += 1

print 'trainSentCount: ', trainSentCount

model = train(verbs)


out = file(OUT_FILE,'w')
for sent in sents:
    pdist = model.prob_classify(extract_posSent_features([sent],False))
    sent = postproc_sents(sent)
    if len(sent) > 0:
        if (pdist.prob('True') >= .5):
            out.write(' '.join([w[0] for w in sent]))
            out.write('\n')
out.flush()
out.close()



            
