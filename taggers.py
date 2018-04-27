import nltk
from nltk.corpus import brown
from nltk.metrics import precision, recall, f_measure
import numpy as np

def most_common_tag(corpus, category):
    tags = [tag for (word, tag) in corpus.tagged_words(categories=category, tagset='universal')]
    return nltk.FreqDist(tags).max()

def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]

def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

def create_confusion_matrix(corpus, category, tagger):
    
    # get a list of the gold standard tags, and the tags set by the tagger.
    gold = tag_list(corpus.tagged_sents(categories=category))
    test = tag_list(apply_tagger(tagger, corpus.tagged_sents(categories=category)))
    
    # create the confusion matrix and return it in a pretty-printed format.
    cm = nltk.ConfusionMatrix(gold, test)
    return cm.pretty_format(sort_by_count=True, show_percents=True, truncate=20)

def evaluate_precision_recall_fmeasure(corpus, category, tagger):
    
    # get a list of the gold standard tags, and the tags set by the tagger.
    gold = set(tag_list(corpus.tagged_sents(categories=category)))
    test = set(tag_list(apply_tagger(tagger, corpus.tagged_sents(categories=category))))
    
    # return the precision and recall of the evaluated model.
    return [precision(gold, test), recall(gold, test), f_measure(gold, test)]

def evaluate_accuracy(tagger, test_set = None):

    # if a test set is specified, evaluate the tagger against it.
    if (test_set):
        return tagger.evaluate(test_set)
    
    # otherwise, evaluate the tagger against the brown corpus.
    else:
        accuracies = []
        for category in brown.categories():
            accuracies.append(tagger.evaluate(brown.tagged_sents(categories=category)))

    # return the average accuracy.
    return np.average(accuracies)

# ============================================================
# Tagger Definitions Begin:
# ============================================================

def default_tagger():
    return nltk.DefaultTagger('NN')

def regex_tagger():
    
    # patterns without comments are self-explanatory and are matching
    # to what is defined in the first element of the tuple.
    patterns = [
         (',$', ','),                        
         (':$', ':'),                        
         ('[.;?!]$', '.'),                   # sentence closers
         ('-', '--'),
         ('have$', 'HV'),                    
         ('having$', 'HVG'),                 
         ('has$', 'HVZ'),                    
         ('be$', 'BE'),
         ('were$', 'BED'),
         ('was$', 'BEDZ'),
         ('being$', 'BEG'),
         ('am$', 'BEM'),
         ('been$', 'BEN'),
         ('is$', 'BEZ'),
         ('do$', 'DO'),
         ('did$', 'DOD'),
         ('does$', 'DOZ'),
         ('a(re|rt)$', 'BER'),              # are or art
         ('.*ing$', 'VBG'),                 # gerunds
         ('.*ed$', 'VBD'),                  # simple past
         ('.*ught$', 'VBD'),                # simple past
         ('.*es$', 'VBZ'),                  # 3rd singular present
         ('.*ould$', 'MD'),                 # modals
         ('.*\'s$', 'NN$'),                 # possessive nouns
         ('.*s$', 'NNS'),                   # plural nouns
         ('.*ly$', 'RB'),                   # adverbs
         ('^-?[0-9]+(.[0-9]+)?$', 'CD'),    # cardinal numbers
         ('^.*(one|two|three|four|five|six|seven|eight|nine|ten)$', 'CD'), # cardinal numbers
    ]
    return nltk.RegexpTagger(patterns, backoff = default_tagger())

def lookup_tagger(words, tagged_words, backoff_tagger = None, num_most_common = None):
    
    if not num_most_common:
        num_most_common = 57340
    
    fd = nltk.FreqDist(words)
    cfd = nltk.ConditionalFreqDist(tagged_words)
    
    most_freq_words = fd.most_common(num_most_common)
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    return nltk.UnigramTagger(model=likely_tags, backoff=backoff_tagger)

def unigram_tagger(train_sents, backoff_tagger = None):
    return nltk.UnigramTagger(train_sents, backoff = backoff_tagger)

def bigram_tagger(train_sents, backoff_tagger = None, cutoff = 0):
    return nltk.BigramTagger(train_sents, backoff = backoff_tagger, cutoff = cutoff)

def trigram_tagger(train_sents, backoff_tagger = None, cutoff = 0):
    return nltk.TrigramTagger(train_sents, backoff = backoff_tagger, cutoff = cutoff)
    
def backoff_model(words, tagged_words, training_set, num_most_common = None):
    
    # define our taggers and their backoff taggers.
    regex_backoff   = regex_tagger()
    lookup_backoff  = lookup_tagger(words, tagged_words, regex_backoff, num_most_common)
    unigram_backoff = unigram_tagger(training_set, lookup_backoff)
    bigram_backoff  = bigram_tagger(training_set, unigram_backoff, 0)
    complete_tagger = trigram_tagger(training_set, bigram_backoff, 4)
    
    # return the model for evaluation.
    return complete_tagger