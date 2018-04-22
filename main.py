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
plot_bar(taggers_used, average_accuracies, "default-only")

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
