from nltk.corpus import wordnet as wn
from qa_howard import get_synsets
from qa_jim import lemmatizer
import nltk
from qa_jim import get_synsets
from qa_jim import tokenize_words
from qa_jim import lemmatize_vb_block
import qa_jim
import re

#
# new_question_text = 'What began to be down over a city ?'
#
# normalized_words = qa_jim.normalize_to_words(new_question_text, 'q')
# q_start_word = normalized_words[0]
# q_normalized_words = normalized_words[1:]  # remove start_word
# key_words, single_vb = qa_jim.lemmatize_vb(q_normalized_words)
# print(key_words)
#
# # sentence = "A group of neighbors didn't begin to be fortunate."
# sentence = "The group of power lines began to be down over a city, and the fourth tree began to be down over the city."
#
# normalized_sent_words = qa_jim.normalize_to_words(sentence)
# sent_word_tag = nltk.pos_tag(tokenize_words(sentence))
# stopwords = set(nltk.corpus.stopwords.words('english'))
# stopwords.remove('from')
# normalized_sent_word_tag = [(word.lower(), tag) for word, tag in
# 									sent_word_tag if word.lower() not in stopwords
# 									and re.search('^[a-z]+', word.lower())]
#
# key_sent_words = qa_jim.lemmatize_vb_with_tag(normalized_sent_word_tag)
# print(key_sent_words)
#
# score = 0
#
# for word in key_words:
# 	if word in key_sent_words:
# 		if word == 'the':
# 			score += 0.5
# 		elif word is single_vb:
# 			score += 1.5
# 		else:
# 			score += 1
# print(score)

print(get_synsets('devour'))