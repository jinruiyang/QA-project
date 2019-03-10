from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk
import re
from nltk.corpus import wordnet as wn
from collections import Counter
from wordnet_loader import load_reshape_wordnet_dict

BE_VERBS = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been'])

lemmatizer = nltk.wordnet.WordNetLemmatizer()


# question in the following code stands for the question_dict object
# a whole row in the question.tsv


def tokenize_sentences(paragraph):
	return [sentence for sentence in nltk.sent_tokenize(paragraph)]


def tokenize_words(sentence):
	return [word for word in nltk.word_tokenize(sentence)]

# def remove_question_words(words):
#     lowered = [w.lower() for w in words]
#     filtered = []
#     question_words = ['?', "\'"]
#
#     for word in lowered:
#         if word not in question_words:
#             filtered.append(word)
#     return filtered

"""
	Find the main_noun and main_verb in the pos_tag list.
	Return the main_noun and main_verb in the given pos_tag.
"""


def words_with_pos_tag(pos_tag, words):
    pos_tags = nltk.pos_tag(words)
    return [words[i] for i in range(len(words)) if pos_tag in pos_tags[i]]


def text_to_sent_words(text):
	sentences = []
	for sentence in nltk.sent_tokenize(text):
		sentences.append(normalize_and_lemmatize(sentence))
	return sentences


def normalize_and_lemmatize(str_of_sentence):
	normalized_words = normalize_to_words(str_of_sentence)
	return lemmatize_vb(normalized_words)


def lemmatize_vb_with_tag(word_tag):
	single_vb = None
	lemmatized_words = []
	vb_count = 0
	for word, tag in word_tag:
		if word == 'spat':
			lemmatized_words.append('spit')
			continue
		if 'VB' in tag:
			if vb_count == 0:
				single_vb = word
			if vb_count == 1:
				single_vb = None
			if word == 'felt':
				lemmatized_words.append('feel')
			if word == 'saw':
				lemmatized_words.append('see')
			if word == 'ate':
				lemmatized_words.append('eat')
			else:
				lemmatized_words.append(lemmatizer.lemmatize(word, 'v'))
		else:
			lemmatized_words.append(word)
	return lemmatized_words, single_vb


def lemmatize_vb(words):
	key_vb = None
	vb_count = 0
	lemmatized_words = []
	for word, tag in nltk.pos_tag(words):
		if word == 'spat':
			lemmatized_words.append('spit')
			continue
		if 'VB' in tag:
			if vb_count == 0:
				key_vb = word
			if vb_count == 1:
				key_vb = None

			if word == 'felt':
				lemmatized_words.append('feel')
			if word == 'saw':
				lemmatized_words.append('see')
			if word == 'ate':
				lemmatized_words.append('eat')
			else:
				lemmatized_words.append(lemmatizer.lemmatize(word, 'v'))
		else:
			lemmatized_words.append(word)

	return lemmatized_words, key_vb

# now we have to remove stopwords of sentences after some keyword detection
def normalize_to_words(str_of_sentence, sent_type='text'):
	words = tokenize_words(str_of_sentence)
	if sent_type == 'q' or True:
		lower_words = [w.lower() for w in words if re.search('^\w', w)]
		return lower_words
	# stopwords = set(nltk.corpus.stopwords.words('english'))
	# stopwords.remove('from')
	# normalized_words = [word for word in words if word not in stopwords and re.search('^\w', word)]
	# return normalized_words


def get_synsets(word):
	word_synsets = wn.synsets(word)
	synonym_synset_ids = []
	hypernym_synset_ids = []
	hyponym_synset_ids = []
	for synset in word_synsets:
		synonym_synset_ids.append(synset.name())
		if synset.hypernyms():
			hypernym_synsets = synset.hypernyms()
			hypernym_synset_ids += [synset.name() for synset in hypernym_synsets]
		if synset.hyponyms():
			hyponym_synsets = synset.hyponyms()
			hyponym_synset_ids += [synset.name() for synset in hyponym_synsets]
	synsets = synonym_synset_ids + hypernym_synset_ids + hyponym_synset_ids
	return synsets


def sub_word_qdep(question, word_index, word_synonym):
	qgraph = question['dep']
	qgraph.nodes[word_index + 1]['word'] = word_synonym


# special past tense can find original form, but doesn't work the other way around
def find_synonym_n_replace(question, key_word_tag, vb_n_synset_dicts, question_words, lemma_word_dict):
	for word, tag in key_word_tag:
		if 'NN' in tag or 'VB' in tag:
			word_synsets_set = set(get_synsets(word))
			if 'VB' in tag:
				synset_word_dict = vb_n_synset_dicts[0]
			if 'NN' in tag:
				synset_word_dict = vb_n_synset_dicts[1]
			for synset_id in synset_word_dict:
				if synset_id in word_synsets_set:
					word_synonym = synset_word_dict[synset_id]
					word_index = question_words.index(lemma_word_dict[word])
					sub_word_qdep(question, word_index, word_synonym)
					question_words[word_index] = word_synonym
					break


def lemmatize_v_n(word_tag):
	lemmatized_word_tag = []
	lemma_word_dict = {}
	for word, tag in word_tag:
		if 'VB' in tag or 'NN' in tag:
			if 'VB' in tag:
				lemmatized_word = lemmatizer.lemmatize(word, 'v')
			if 'NN' in tag:
				lemmatized_word = lemmatizer.lemmatize(word)
			lemmatized_word_tag.append((lemmatized_word, tag))
			lemma_word_dict[lemmatized_word] = word
	return lemmatized_word_tag, lemma_word_dict


def replace_synonym(question, reshape_verb_dict, reshape_noun_dict):
	verb_synset_word_dict = reshape_verb_dict[question['sid']]
	noun_synset_word_dict = reshape_noun_dict[question['sid']]
	stopwords = set(nltk.corpus.stopwords.words('english'))
	question_words = tokenize_words(question['text'])
	question_word_tag = nltk.pos_tag(question_words)
	lemmatized_word_tag, lemma_word_dict = lemmatize_v_n(question_word_tag)
	key_word_tag = [(word, tag) for word, tag in lemmatized_word_tag
					if re.search('^[a-z]+$', word) and word not in stopwords]
	find_synonym_n_replace(question, key_word_tag, (verb_synset_word_dict, noun_synset_word_dict), question_words, lemma_word_dict)
	new_question_text = ' '.join(question_words)
	return new_question_text


# originally called compare_sentence, parameter question changed to question_text
def match_sent_from_q(question, sentences):
	reshape_verb_dict = load_reshape_wordnet_dict('v')
	reshape_noun_dict = load_reshape_wordnet_dict('n')

	if question['difficulty'] == 'Hard' and question['sid'] in {**reshape_verb_dict, **reshape_noun_dict}:
		new_question_text = replace_synonym(question, reshape_verb_dict, reshape_noun_dict)
		print('question before sub:')
		print(question['text'])
	else:
		new_question_text = question['text']

	max_match = 0
	matched_sentence = sentences[0], 0  # better return a sentence in the text

	normalized_words = normalize_to_words(new_question_text, 'q')
	q_start_word = normalized_words[0]
	q_normalized_words = normalized_words[1:]  # remove start_word
	key_words, key_vb = lemmatize_vb(q_normalized_words)
	if key_words[-1] == 'about':
		return sentences[0]

	print("{Question_Text}: " + new_question_text)

	have_to_flag = False

	index = 0
	matched_index = 0
	for sentence in sentences:
		score = 0
		if have_to_flag:
			score += 0.5
		normalized_sent_words = normalize_to_words(sentence)

		# detection of key stopwords from here
		if q_start_word == 'why':
			if 'because' in normalized_sent_words:
				score += 1
			if key_vb:
				if 'has to ' + key_vb in sentence or 'have to ' + key_vb in sentence or 'had to ' + key_vb in sentence:
					have_to_flag = True

		# let's try pos_tag first, store the pairs(word, tag), remove stopwods, then lemmatize.
		sent_word_tag = nltk.pos_tag(tokenize_words(sentence))
		stopwords = set(nltk.corpus.stopwords.words('english'))
		stopwords.remove('from')
		normalized_sent_word_tag = [(word.lower(), tag) for word, tag in sent_word_tag if word not in stopwords]

		key_sent_words, single_vb = lemmatize_vb_with_tag(normalized_sent_word_tag)

		for word in key_words:
			if word in key_sent_words:
				if word == 'the':
					score += 0.5
				elif word is key_vb:
				# should use the single word in sentence.
					score += 1.5
				else:
					score += 1

		if score > max_match:
			matched_sentence = sentence
			max_match = score
			matched_index = index

		index += 1

	if 'what' in q_start_word and 'say' in key_words:
		if Counter(matched_sentence)['\"'] == 1:
			matched_quote = matched_sentence
			for i in range(matched_index + 1, len(sentences)):
				matched_quote += ' '
				matched_quote += sentences[i]
				if Counter(sentences[i])['\"'] > 0:
					return matched_quote.lower()

	return matched_sentence


def get_answer_with_overlap(question, story):
	if 'Sch' in question['type']:
		matched_sentence = match_sent_from_q(question, tokenize_sentences(story['sch']))


		# matched_sch_dep = story['sch_dep'][matched_index]
		# print(matched_sch_dep)

		# answer = find_the_answer(matched_sentence,question['text'])
	else:
		matched_sentence = match_sent_from_q(question, tokenize_sentences(story['text']))

		# matched_story_dep = story['story_dep'][matched_index]
		# print(matched_story_dep)

		# answer = find_the_answer(matched_sentence,question['text'])
	answer = matched_sentence
	return answer, matched_sentence


""" 
	Chunking on the high recall answer (1 sentence only).
	If we have time use stanford pos tagging (higher precision).
"""
def get_answer_with_chunck(question, matched_sentence, raw_sent_answer):
	"""
	: param question: the question dict of a row in question.tsv
	: param matched_sentence: the sentence in sch or story texts that has the most overlap
	: return str: the chunked matched sentence
	"""
	answer = ""
	question_sents = get_sentences(question["text"])
	# start word of the question, ex: what, why, where, when, who
	q_start_word = question_sents[0][0][0].lower()

	if q_start_word in ("when","where","why"):
		sentence_words = nltk.word_tokenize(matched_sentence)
		sentence_word_tag = nltk.pos_tag(sentence_words)
		answer_tree = chunk.find_candidates([sentence_word_tag],chunker,q_start_word)
		# if we found the answer
		if answer_tree:
			answer = " ".join([token[0] for token in answer_tree[0].leaves()])
		else:
			print("Its raw_sent_answer:")
			answer = raw_sent_answer
	else:
		answer = raw_sent_answer
	return answer


def get_answer(question, story):
	print('=========================================')
	raw_sent_answer, matched_sentence = get_answer_with_overlap(question, story)
	# print(normalize_and_lemmatize(question['text']))
	# if 'Sch' in question['type']:
	# 	print(story['sch'])
	# 	print(text_to_sent_words(story['sch']))
	# else:
	# 	print(story['text'])
	# 	print(text_to_sent_words(story['text']))
	# print(matched_sentence)
	return raw_sent_answer
	# answer = get_answer_with_chunck(question, matched_sentence, raw_sent_answer)
	print("{Sentence}:", matched_sentence)
	print("{Answer}:", answer)
	print("\n")
	"""
	:param question: dict
	:param story: dict
	:return: str


	question is a dictionary with keys:
		dep -- A list of dependency graphs for the question sentence.
		par -- A list of constituency parses for the question sentence.
		text -- The raw text of story.
		sid --  The story id.
		difficulty -- easy, medium, or hard
		type -- whether you need to use the 'sch' or 'story' versions
				of the .


	story is a dictionary with keys:
		story_dep -- list of dependency graphs for each sentence of
					the story version.
		sch_dep -- list of dependency graphs for each sentence of
					the sch version.
		sch_par -- list of constituency parses for each sentence of
					the sch version.
		story_par -- list of constituency parses for each sentence of
					the story version.
		sch --  the raw text for the sch version.
		text -- the raw text for the story version.
		sid --  the story id


	"""
	#if(question['type'] == "sch"):
	#    text = story['sch']
	###     Your Code Goes Here         ###

	#answer = "whatever you think the answer is"

	###     End of Your Code         ###

	return answer



#############################################################
###     Dont change the code in this section
#############################################################


class QAEngine(QABase):
	@staticmethod
	# override the answer_question function of QAEngine class in base.py
	def answer_question(question, story):
		answer = get_answer(question, story)
		return answer


def run_qa(evaluate=False):
	QA = QAEngine(evaluate=evaluate)
	QA.run()
	QA.save_answers()

#############################################################


def run_qa_with_score(evaluate=False):
	QA = QAEngine(evaluate=evaluate)
	QA.run_score('Hard')
	# QA.run_score(set(['what']))

def main():

	# set evaluate to True/False depending on whether or
	# not you want to run your system on the evaluation
	# data. Evaluation data predictions will be saved
	# to hw6-eval-responses.tsv in the working directory.
	# run_qa(evaluate=False)
	# You can uncomment this next line to evaluate your
	# answers, or you can run score_answers.py
	# score_answers()
	run_qa_with_score()

if __name__ == "__main__":
	main()