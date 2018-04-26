import nltk
from nltk.corpus import brown
import taggers

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
regex_tagger = taggers.regex_tagger()
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
lookup_tagger   = taggers.lookup_tagger(all_words, all_tagged_words, default_tagger)
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