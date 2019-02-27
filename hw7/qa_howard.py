from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk
import chunk
import re
from nltk.corpus import wordnet as wn

BE_VERBS = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been'])

lemmatizer = nltk.wordnet.WordNetLemmatizer()

chunker = chunk.build_chunker()

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


def lemmatize_vb(words):
	key_vb = None
	vb_count = 0
	lemmatized_words = []
	for word, tag in nltk.pos_tag(words):
		if 'VB' in tag:
			if vb_count == 0:
				key_vb = word
			if vb_count == 1:
				key_vb = None

			if word == 'felt':
				lemmatized_words.append('feel')
			else:
				lemmatized_words.append(lemmatizer.lemmatize(word, 'v'))
		else:
			lemmatized_words.append(word)

	return lemmatized_words, key_vb

# now we have to remove stopwords of sentences after some keyword detection
def normalize_to_words(str_of_sentence, sent_type='text'):
	words = tokenize_words(str_of_sentence)
	if sent_type == 'q' or True:
		lower_words = [w.lower() for w in words if re.search('^\w', w) and w not in BE_VERBS]
		return lower_words
	# stopwords = set(nltk.corpus.stopwords.words('english'))
	# stopwords.remove('from')
	# normalized_words = [word for word in words if word not in stopwords and re.search('^\w', word)]
	# return normalized_words


# special past tense can find original form, but doesn't work the other way around
def find_synonyms(word):
	return [str(synset.name()).split('.')[0] for synset in wn.synsets(word)]


# originally called compare_sentence, parameter question changed to question_text
def match_sent_from_q(question_text, sentences):
	max_match = 0
	matched_sentence = sentences[0]  # better return a sentence in the text

	normalized_words = normalize_to_words(question_text, 'q')
	q_start_word = normalized_words[0]
	key_words, key_vb = lemmatize_vb(normalized_words)
	if key_words[-1] == 'about':
		return sentences[0]

	print("{Question_Text}: " + question_text)

	have_to_flag = False
	for sentence in sentences:
		if have_to_flag:
			return sentence
		score = 0
		normalized_sent_words = normalize_to_words(sentence)

		# detection of key stopwords from here
		if q_start_word == 'why':
			if 'because' in normalized_sent_words:
				score += 1
			if key_vb:
				if 'has to ' + key_vb in sentence or 'have to ' + key_vb in sentence or 'had to ' + key_vb in sentence:
					have_to_flag = True

		# remove stopwords after detection
		stopwords = set(nltk.corpus.stopwords.words('english'))
		stopwords.remove('from')
		normalized_sent_words = [word for word in normalized_sent_words if word not in stopwords]

		key_sent_words = set([word.lower() for word in lemmatize_vb(normalized_sent_words)[0]])
		for word in key_words:
			if word in key_sent_words:
				if word == 'the':
					score += 0.5
				else:
					score += 1

		if score > max_match:
			matched_sentence = sentence
			max_match = score

	return matched_sentence


def find_the_answer(matched_sentence,question):
	ps = PorterStemmer()
	string = []
	word_answer = tokenize_words(matched_sentence)
	word_question = tokenize_words(question)
	clue = word_question[0].lower()
	word_answer = nltk.pos_tag(word_answer)

	words = normalize_to_words(question)

	#define the main_verb (the verb appeared in question)
	main_verb = [w for (w,a) in words if a == 'v']
	print(main_verb)
	# Get subject, no object
	if clue == 'what':
		#print(word_answer)
		for index, item in enumerate(word_answer):
			if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
				#find out sth is/are/are/was/were doing  OR sth is/are/are/was/were done,
				#then pick out the part before is/are/are/was/were as answer
				if 'VB' in word_answer[index-1][1]:
					for item in word_answer[:index-1]:
						#string = string+item[0]+" "
						string.append(item[0])
					#print("T1 Find the subject of -ing or -ed:", string)
				# find out sth/sb do sth, find out sth -ing -ed as the adjective
				# then pick out the part after the verb as answer
				else:
					for item in word_answer[index+1:]:
						#string = string+item[0]+" "
						string.append(item[0])
					#print("T2 Find the object", string)
	if clue == 'why':
		found = False
		for item in tokenize_words(matched_sentence):
			if item == 'because':
				found = True
			if 'because' not in item and found == True:
				#string = string+item+" "
				string.append(item)
	#print("string: "+string)

	if clue == 'who':
		#print(word_answer)
		for index, item in enumerate(word_answer):
			if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
				# find out sth/sb do sth
				# then pick out the part before the verb as answer
				for item in word_answer[:index]:
					if 'VB' not in item[1]:
						#string = string+item[0]+" "
						string.append(item[0])
				#print("T3 Find the subject who type question", string)

	if len(string) == 0:
		return matched_sentence

	return " ".join(string)


def get_answer_with_overlap(question, story):
	answer = ""
	matched_sentence = ""
	if 'Sch' in question['type']:
		matched_sentence = match_sent_from_q(question['text'], tokenize_sentences(story['sch']))

		# answer = find_the_answer(matched_sentence,question['text'])
	else:
		matched_sentence = match_sent_from_q(question['text'], tokenize_sentences(story['text']))

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
	q_start_words = set(['what', 'when', 'where', 'who', 'why', 'how', 'did', 'had'])
	# QA.run_score(q_start_words)
	QA.run_score(set(['why']))

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