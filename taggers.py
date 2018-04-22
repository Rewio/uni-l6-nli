import nltk

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
    return cm.pretty_format(sort_by_count=True, show_percents=True, truncate=10)

def default_tagger():
    return nltk.DefaultTagger('NN')

def regex_tagger(backoff_tagger = None):
    patterns = [
         (r'.*ing$', 'VBG'),               # gerunds
         (r'.*ed$', 'VBD'),                # simple past
         (r'.*es$', 'VBZ'),                # 3rd singular present
         (r'.*ould$', 'MD'),               # modals
         (r'.*\'s$', 'NN$'),               # possessive nouns
         (r'.*s$', 'NNS'),                 # plural nouns
         (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
         (r'.*', 'NN')                     # nouns (default)
    ]
    return nltk.RegexpTagger(patterns, backoff=None)

def lookup_tagger(words, tagged_words, backoff_tagger = None):
    fd = nltk.FreqDist(words)
    cfd = nltk.ConditionalFreqDist(tagged_words)
    most_freq_words = fd.most_common(100)
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    return nltk.UnigramTagger(model=likely_tags, backoff=backoff_tagger)

def unigram_tagger(train_sents, backoff_tagger = None):
    return nltk.UnigramTagger(train_sents, backoff=backoff_tagger)

def bigram_tagger(train_sents, backoff_tagger = None):
    return nltk.BigramTagger(train_sents, backoff=backoff_tagger)

def trigram_tagger(train_sents, backoff_tagger = None):
    return nltk.TrigramTagger(train_sents, backoff=backoff_tagger, cutoff=2)
    
def evaluate_backoff_model(corpus):
    
    # get the words and the tagged words, used by the lookup tagger.
    words = corpus.words(categories='news')
    tagged_words = corpus.tagged_words(categories='news')
    
    # get the tagged sentences to create our train and test sets, used by the unigram and bigram taggers.
    tagged_text = corpus.tagged_sents(categories='news')
    size = int(len(tagged_text) * 0.9)
    train_sents = tagged_text[:size]
    test_sents = tagged_text[size:]
    
    # define our taggers and their backoff taggers.
    t0 = default_tagger()
    t1 = regex_tagger(t0)
    t2 = lookup_tagger(words, tagged_words, t1)
    t3 = unigram_tagger(train_sents, t2)
    t4 = bigram_tagger(train_sents, t3)
    t5 = trigram_tagger(train_sents, t4)
    
    # evaluate the test set using our tagger backoff model.
    return t5.evaluate(test_sents)




#brown_tagged_sents = brown.tagged_sents(categories='news')
#brown_sents = brown.sents(categories='news')
#
#tokens = word_tokenize("This is a series of tokens that will be tokenized")
#
## ==================================================
## Default Tagger:
## ==================================================
#
#default_tagger = nltk.DefaultTagger('NN')
#print(default_tagger.tag(tokens))
#default_tagger.evaluate(brown_tagged_sents)
#
## ==================================================
## Regular Expression Tagger:
## ==================================================
#
#patterns = [
#     (r'.*ing$', 'VBG'),               # gerunds
#     (r'.*ed$', 'VBD'),                # simple past
#     (r'.*es$', 'VBZ'),                # 3rd singular present
#     (r'.*ould$', 'MD'),               # modals
#     (r'.*\'s$', 'NN$'),               # possessive nouns
#     (r'.*s$', 'NNS'),                 # plural nouns
#     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
#     (r'.*', 'NN')                     # nouns (default)
#]
#regexp_tagger = nltk.RegexpTagger(patterns)
#regexp_tagger.tag(brown_sents[3])
#regexp_tagger.evaluate(brown_tagged_sents)
#
## ==================================================
## Lookup Tagger:
## ==================================================
#
#fd = nltk.FreqDist(brown.words(categories='news'))
#cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
#most_freq_words = fd.most_common(100)
#likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
#baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
#baseline_tagger.evaluate(brown_tagged_sents)
#
## ==================================================
## Unigram Tagger:
## ==================================================
#
#unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
#unigram_tagger.tag(brown_sents[2007])
#unigram_tagger.evaluate(brown_tagged_sents)
#
#
#size = int(len(brown_tagged_sents) * 0.9)
#train_sents = brown_tagged_sents[:size]
#test_sents = brown_tagged_sents[size:]
#unigram_tagger = nltk.UnigramTagger(train_sents)
#unigram_tagger.evaluate(test_sents)
#
## ==================================================
## Bigram Tagger:
## ==================================================
#
#bigram_tagger = nltk.BigramTagger(test_sents)
#bigram_tagger.tag(brown_sents[2007])
#unseen_sent = brown_sents[4203]
#bigram_tagger.tag(unseen_sent)
#bigram_tagger.evaluate(test_sents)