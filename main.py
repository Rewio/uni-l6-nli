import nltk
from nltk.corpus import brown
import numpy as np

import taggers
from plot import plot_bar

taggers_used = []
accuracies = []
average_accuracies = []

# find the most common tag used in each of the brown categories.
most_common_tags = [taggers.most_common_tag(brown, category) for category in brown.categories()]

# ============================================================
# Default Tagger:
# ============================================================

# append the tagger name to our taggers list for plotting.
taggers_used.append("Default")

# instantiate the default tagger with the default tag as Noun.
default_tagger = nltk.DefaultTagger('NN')

# evaluate the tagger against each category in the brown corpus, storing the accuracies.
for category in brown.categories():
    accuracies.append(default_tagger.evaluate(brown.tagged_sents(categories=category)))
    
# get the average accuracy of the model and append this to the average_accuracies list for plotting.
average_accuracies.append(np.average(accuracies))

# plot the accuracy of the taggers and their accuracies thus far on a bar chart.
plot_bar(taggers_used, average_accuracies, "default")

# print a pretty-printed confusion matrix for the default tagger against the news category of the brown corpus.
print("\n" + taggers.create_confusion_matrix(brown, "news", default_tagger))

# ============================================================
# Regex Tagger:
# ============================================================

# this code block follows the same structure as the default tagger code block, comments can be found there.

taggers_used.append("Regex")
regex_tagger = taggers.regex_tagger()
accuracies = []
for category in brown.categories():
    accuracies.append(regex_tagger.evaluate(brown.tagged_sents(categories=category)))
average_accuracies.append(np.average(accuracies))
plot_bar(taggers_used, average_accuracies, "regex")
print("\n" + taggers.create_confusion_matrix(brown, "news", regex_tagger))

# ============================================================
# Lookup Tagger:
# ============================================================

# instantiate our containers.
all_words        = []
all_tagged_words = []

# iterate over the brown corpus tracking words and tagged words, to train our lookup tagger.
for category in brown.categories():
    for word in brown.words(categories=category):
        all_words.append(word)
    for tagged_word in brown.tagged_words(categories=category):
        all_tagged_words.append(tagged_word)
        
# same as the previous two code blocks.    
    
taggers_used.append("Lookup")        
lookup_tagger = taggers.lookup_tagger(all_words, all_tagged_words, default_tagger)
accuracies = []
for category in brown.categories():
    accuracies.append(lookup_tagger.evaluate(brown.tagged_sents(categories=category)))
average_accuracies.append(np.average(accuracies))
plot_bar(taggers_used, average_accuracies, "lookup")
print("\n" + taggers.create_confusion_matrix(brown, "news", lookup_tagger))

# ============================================================
# Unigram Tagger:
# ============================================================

all_tagged_sentences = []
for category in brown.categories():
    for sentence in brown.tagged_sents(categories=category):
        all_tagged_sentences.append(sentence)

taggers_used.append("Unigram")  
size = int(len(all_tagged_sentences) * 0.8)
unigram_tagger = taggers.unigram_tagger(all_tagged_sentences[:size], default_tagger)
accuracies = []
for category in brown.categories():
    accuracies.append(unigram_tagger.evaluate(brown.tagged_sents(categories=category)))
plot_bar(taggers_used, average_accuracies, "lookup")
print("\n" + taggers.create_confusion_matrix(brown, "news", unigram_tagger))