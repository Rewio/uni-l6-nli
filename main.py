import nltk
from nltk.corpus import brown

import numpy as np
import matplotlib.pyplot as plt

from taggers import most_common_tag, create_confusion_matrix, evaluate
from plot import plot_bar

taggers = []
accuracies = []
average_accuracies = []

# find the most common tag used in each of the brown categories.
most_common_tags = [most_common_tag(brown, category) for category in brown.categories()]

# setup a defult tagger using noun as the default, and test it against each category in the brown corpus.
default_tagger = nltk.DefaultTagger('NN')
taggers.append("Default")
for category in brown.categories():
    accuracies.append(default_tagger.evaluate(brown.tagged_sents(categories=category)))
average_accuracies.append(np.average(accuracies))
print(create_confusion_matrix(brown, "news", default_tagger))

plot_bar(taggers, average_accuracies, "default-only")