import nltk
import numpy as np
from nltk.corpus import brown
import taggers
import plot

# find the most common tag used in each of the brown categories.
most_common_tags = [taggers.most_common_tag(brown, category) for category in brown.categories()]

# ============================================================
# Default Tagger:
# ============================================================

# instantiate the default tagger then evaluate it.
default_tagger   = nltk.DefaultTagger('NN')
default_accuracy = taggers.evaluate_accuracy(default_tagger)

# ============================================================
# Regex Tagger:
# ============================================================

# instantiate the regex tagger then evaluate it.
regex_tagger   = taggers.regex_tagger()
regex_accuracy = taggers.evaluate_accuracy(regex_tagger)

# ============================================================
# Lookup Tagger:
# ============================================================

# instantiate our containers.
all_words, all_tagged_words = [], []

# iterate over the brown corpus tracking words and tagged words, to train our lookup tagger.
for category in brown.categories():
    for word in brown.words(categories=category):
        all_words.append(word)
    for tagged_word in brown.tagged_words(categories=category):
        all_tagged_words.append(tagged_word)

# train the lookup tagger and evaluate it's accuracy.
lookup_tagger   = taggers.lookup_tagger(all_words, all_tagged_words, default_tagger, 2000)
lookup_accuracy = taggers.evaluate_accuracy(lookup_tagger)

# ============================================================
# Unigram Tagger:
# ============================================================

# collect all of the tagged sentences across the entire brown corpus.
all_tagged_sentences = []
for category in brown.categories():
    for sentence in brown.tagged_sents(categories=category):
        all_tagged_sentences.append(sentence)

# split our tagged sentences into a training and testing set.
size      = int(len(all_tagged_sentences) * 0.8)
train_set = all_tagged_sentences[:size]
test_set  = all_tagged_sentences[size:]

# train the unigram tagger using the train set, then evaluate it against the test set.
unigram_tagger   = taggers.unigram_tagger(train_set, default_tagger)
unigram_accuracy = taggers.evaluate_accuracy(unigram_tagger, test_set)

# ============================================================
# Bigram Tagger:
# ============================================================

bigram_tagger   = taggers.bigram_tagger(train_set, default_tagger, 0)
bigram_accuracy = taggers.evaluate_accuracy(bigram_tagger, test_set)

# ============================================================
# Trigram Tagger:
# ============================================================

trigram_tagger   = taggers.trigram_tagger(train_set, default_tagger, 0)
trigram_accuracy = taggers.evaluate_accuracy(trigram_tagger, test_set)

# ============================================================
# Backoff Model:
# ============================================================

backoff_model    = taggers.backoff_model(all_words, all_tagged_words, train_set)
backoff_accuracy = taggers.evaluate_accuracy(backoff_model, test_set)

# ============================================================
# Plotting and Metrics:
# ============================================================

plot.plot_bar(["Default", "Regex", "Lookup", "Unigram", "Bigram", "Trigram", "Backoff"],
              [default_accuracy, regex_accuracy, lookup_accuracy, unigram_accuracy,
               bigram_accuracy, trigram_accuracy, backoff_accuracy], "all-taggers")
    
prec_rec_f1 = taggers.evaluate_precision_recall_fmeasure(brown, "news", backoff_model)
cm = taggers.create_confusion_matrix(brown, "news", backoff_model)

# ============================================================
# Task 2:
# ============================================================

# declare the sizes we wish to use when training, and the container with which we will record their accuracies.
sizes = 2 ** np.arange(16)
accuracies = []

# iterate over each of the sizes...
for size in sizes:
    
    # declare the training and testing set, based on this iterations size.
    training_set = all_tagged_sentences[:size]
    testing_set  = all_tagged_sentences[size:]
    
    # train our model using the training set, then evaluate it against the test set.
    model = taggers.unigram_tagger(training_set, default_tagger)
    accuracies.append(taggers.evaluate_accuracy(model, testing_set))
    
# finally plot the accuracies against the sizes using a standard line graph.
plot.plot(sizes, accuracies, "task-2")