from qa_engine.score_answers import main as score_answers
# from qa_engine.type_answer import main as type_answer
from qa_engine.base import QABase
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk
import chunk
import re, os
# import dependency
from nltk.corpus import wordnet as wn
from collections import Counter
from wordnet_loader import load_reshape_wordnet_dict
from wordnet_loader import load_reshape_wordnet_dict_2

BE_VERBS = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been'])

lemmatizer = nltk.wordnet.WordNetLemmatizer()
# To record question type for type_answer to  record only certain types of question
q_start_list = []
# To record entity counts for each story, key:sid, value:list of tuples with the most populor n entities
story_entities = {}
# If want to see certain types of question in the file, set to True and change target_qstart
verbose = False
target_qstart = "where"



# chunker = chunk.build_chunker()

# question in the following code stands for the question_dict object
# a whole row in the question.tsv
def get_sentences(text):
	sentences = nltk.sent_tokenize(text)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]

	return sentences


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


def lemmatize_vb_block(word, lemmatized_words):
	if word == 'felt':
		lemmatized_words.append('feel')
	if word == 'saw':
		lemmatized_words.append('see')
	if word == 'ate':
		lemmatized_words.append('eat')
	if word == 'occurring':
		lemmatized_words.append('occur')
	else:
		lemmatized_verb = lemmatizer.lemmatize(word, 'v')
		# if doesn't change form we can cut ing and ed
		if lemmatized_verb == word:
			if word[-3:] == 'ing':
				lemmatized_words.append(word[:-3])
			elif word[-2:] == 'ed':
				lemmatized_words.append(word[:-2])
			else:
				lemmatized_words.append(lemmatized_verb)
		else:
			lemmatized_words.append(lemmatized_verb)


def lemmatize_vb_with_tag(word_tag):
	lemmatized_words = []
	for word, tag in word_tag:
		if word == 'spat':
			lemmatized_words.append('spit')
			continue
		if 'VB' in tag:
			lemmatize_vb_block(word, lemmatized_words)
		else:
			lemmatized_words.append(word)
	return lemmatized_words


def lemmatize_vb(words):
	single_vb = None
	vb_count = 0
	lemmatized_words = []
	for word, tag in nltk.pos_tag(words):
		if word == 'spat':
			lemmatized_words.append('spit')
			continue
		if 'VB' in tag:
			if vb_count == 0:
				single_vb = word
			if vb_count == 1:
				single_vb = None

			lemmatize_vb_block(word, lemmatized_words)
		else:
			lemmatized_words.append(word)

	return lemmatized_words, single_vb


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
	#print(qgraph.nodes[word_index + 1])


# special past tense can find original form, but doesn't work the other way around
def find_synonym_n_replace(question, key_word_tag, vb_n_synset_dicts, question_words, lemma_word_dict):
	print(key_word_tag)
	for word, tag in key_word_tag:
		if 'NN' in tag or 'VB' in tag:
			word_synsets_set = set(get_synsets(word))
			synset_word_dict = {}
			if 'VB' in tag:
				synset_word_dict = vb_n_synset_dicts[0]
			if 'NN' in tag:
				synset_word_dict = vb_n_synset_dicts[1]
			print(word)
			print(get_synsets(word))
			print(synset_word_dict)
			for synset_id in synset_word_dict:
				if synset_id in word_synsets_set:
					print(synset_id)
					word_synonym = synset_word_dict[synset_id]
					print(word_synonym)
					word_index = question_words.index(lemma_word_dict[word])
					sub_word_qdep(question, word_index, word_synonym)
					question_words[word_index] = word_synonym
					break



def lemmatize_v_n(word_tag):
	lemmatized_word_tag = []
	lemma_word_dict = {}
	for word, tag in word_tag:
		lemmatized_word = ''
		if 'VB' in tag or 'NN' in tag:
			if 'VB' in tag:
				if word == 'felt':
					lemmatized_word = 'feel'
				elif word == 'saw':
					lemmatized_word = 'see'
				elif word == 'ate':
					lemmatized_word = 'eat'
				elif word == 'occurring':
					lemmatized_word = 'occur'
				else:
					lemmatized_verb = lemmatizer.lemmatize(word, 'v')
					# if doesn't change form we can cut ing and ed
					if lemmatized_verb == word:
						if word[-3:] == 'ing':
							lemmatized_word = word[:-3]
						elif word[-2:] == 'ed':
							lemmatized_word = word[:-2]
						else:
							lemmatized_word = lemmatized_verb
					else:
						lemmatized_word = lemmatized_verb
			if 'NN' in tag:
				lemmatized_word = lemmatizer.lemmatize(word)
		else:
			lemmatized_word = word
		lemmatized_word_tag.append((lemmatized_word, tag))
		lemma_word_dict[lemmatized_word] = word
	return lemmatized_word_tag, lemma_word_dict

def get_postag_from_qgraph(qgraph):
	word_tag = []
	size = len(qgraph.nodes)
	for i in range(1, size):
		word, tag = qgraph.nodes[i]['word'], qgraph.nodes[i]['tag']
		word_tag.append((word, tag))
	return word_tag


def valid_words(graph_word_tag, question_word_tag):
		for (word1, tag1), (word2, tag2) in zip(graph_word_tag, question_word_tag):
			if word1 != word2:
				return False
		return True

def replace_synonym(question, reshape_verb_dict, reshape_noun_dict):

	verb_synset_word_dict = reshape_verb_dict[question['sid']]
	noun_synset_word_dict = reshape_noun_dict[question['sid']]
	stopwords = set(nltk.corpus.stopwords.words('english'))
	question_words = tokenize_words(question['text'])
	graph_word_tag = get_postag_from_qgraph(question['dep'])
	question_word_tag = nltk.pos_tag(question_words)
	if len(graph_word_tag) == len(question_word_tag) and valid_words(graph_word_tag, question_word_tag):
		question_word_tag = graph_word_tag
	print(question_word_tag)
	lemmatized_word_tag, lemma_word_dict = lemmatize_v_n(question_word_tag)
	print(lemmatized_word_tag)
	key_word_tag = [(word, tag) for word, tag in lemmatized_word_tag
					if re.search('^[a-z]+$', word) and word not in stopwords]
	find_synonym_n_replace(question, key_word_tag, (verb_synset_word_dict, noun_synset_word_dict), question_words, lemma_word_dict)
	new_question_text = ' '.join(question_words)
	return new_question_text

def match_synsets(synsets_1, synsets_2):
	set_1 = set(synsets_1)
	set_2 = set(synsets_2)
	for synset in synsets_1:
		if synset in set_2:
			return True
	for synset in synsets_2:
		if synset in set_1:
			return True
	return False


def replace_sent_synonym_in_q(lemmatized_word_tag, question):
	replace_dict = {}

	stopwords = set(nltk.corpus.stopwords.words('english'))
	question_words = tokenize_words(question['text'])
	graph_word_tag = get_postag_from_qgraph(question['dep'])
	question_word_tag = nltk.pos_tag(question_words)
	if len(graph_word_tag) == len(question_word_tag) and valid_words(
			graph_word_tag, question_word_tag):
		question_word_tag = graph_word_tag
	print("question_word_tag:{}".format(question_word_tag))
	# print(question_word_tag)
	q_lemmatized_word_tag, q_lemma_word_dict = lemmatize_v_n(question_word_tag)
	# print(lemmatized_word_tag)
	key_word_tag = [(word, tag) for word, tag in q_lemmatized_word_tag
					if re.search('^[a-z]+$', word) and word not in stopwords]
	print("q_key_word_tag:{}".format(key_word_tag))
	reshape_verb_dict = load_reshape_wordnet_dict_2('v')
	reshape_noun_dict = load_reshape_wordnet_dict_2('n')

	verb_synset_word_dict = reshape_verb_dict[question['sid']]
	noun_synset_word_dict = reshape_noun_dict[question['sid']]

	# we only have verbs and nouns in lemmatized_word_tag

	for word, tag in lemmatized_word_tag:
		if 'VB' in tag:
			if word in verb_synset_word_dict:
				word_synset = verb_synset_word_dict[word]
				for q_word, q_tag in key_word_tag:
					q_word_synsets = set(get_synsets(q_word))
					if q_word != word and word_synset in q_word_synsets:
						replace_dict[q_word] = word
			else:
				for q_word, q_tag in key_word_tag:
					if match_synsets(get_synsets(q_word), get_synsets(word)):
						replace_dict[q_word] = word
		if 'NN' in tag:
			if word in noun_synset_word_dict:
				word_synset = noun_synset_word_dict[word]
				for q_word, q_tag in key_word_tag:
					q_word_synsets = set(get_synsets(q_word))
					if 'NN' in q_tag and word_synset in q_word_synsets:
						replace_dict[q_word] = word
			else:
				for q_word, q_tag in key_word_tag:
					if match_synsets(get_synsets(q_word), get_synsets(word)):
						replace_dict[q_word] = word
	key_q_words = []
	for word, tag in key_word_tag:
		if replace_dict.get(word):
			new_word = replace_dict[word]
			key_q_words.append(new_word)
		else:
			key_q_words.append(word)

	return key_q_words, replace_dict

# originally called compare_sentence, parameter question changed to question_text
def match_sent_from_q(question, sentences, story_deps):
	# reshape_verb_dict = load_reshape_wordnet_dict('v')
	# reshape_noun_dict = load_reshape_wordnet_dict('n')
	#
	# if question['difficulty'] == 'Hard' and question['sid'] in {**reshape_verb_dict, **reshape_noun_dict}:
	# 	new_question_text = replace_synonym(question, reshape_verb_dict,
	# 										reshape_noun_dict)
	# 	print('question before sub:')
	# 	print(question['text'])
	# else:
	new_question_text = question['text']

	max_match = 0
	matched_sentence = sentences[0]  # better return a sentence in the text

	normalized_words = normalize_to_words(new_question_text, 'q')
	q_start_word = normalized_words[0]
	q_normalized_words = normalized_words[1:]  # remove start_word
	key_words, single_vb = lemmatize_vb(q_normalized_words)
	if key_words[-1] == 'about':
		return sentences[0], story_deps[0]

	print("{Question_Text}: " + new_question_text)

	have_to_flag = False

	index = 0
	match_index = 0
	sub_record = [] # list of dictionaries of substitutions
	print(sentences)
	for sentence in sentences:

		# if question['text']
		if have_to_flag:
			return sentence, story_deps[match_index]
		score = 0
		normalized_sent_words = normalize_to_words(sentence)

		# detection of key stopwords from here
		if q_start_word == 'why':
			if 'because' in normalized_sent_words:
				score += 1
			if single_vb:
				if 'has to ' + single_vb in sentence or 'have to ' + single_vb in sentence or 'had to ' + single_vb in sentence:
					have_to_flag = True

		# remove stopwords after detection
		# stopwords = set(nltk.corpus.stopwords.words('english'))
		# stopwords.remove('from')
		# normalized_sent_words = [word for word in normalized_sent_words if word not in stopwords]
		#
		# key_sent_words = set([word.lower() for word in lemmatize_vb(normalized_sent_words)[0]])

		# let's try pos_tag first, store the pairs(word, tag), remove stopwods, then lemmatize.


		if len(story_deps) == len(sentences):
			sent_graph_word_tag = get_postag_from_qgraph(story_deps[index])
		else:
			sent_graph_word_tag = []

		sent_word_tag = nltk.pos_tag(tokenize_words(sentence))
		if len(sent_word_tag) == len(sent_graph_word_tag) and valid_words(sent_graph_word_tag, sent_word_tag):
			sent_word_tag = sent_graph_word_tag

		print("sent_word_tag: {}".format(sent_word_tag))

		stopwords = set(nltk.corpus.stopwords.words('english'))
		stopwords.remove('from')
		normalized_sent_word_tag = [(word.lower(), tag) for word, tag in
									sent_word_tag if word.lower() not in stopwords
									and re.search('^[a-z]+', word.lower())]
		lemmatized_sent_word_tag, lemma_word_dict = lemmatize_v_n(normalized_sent_word_tag)  #
		print("lemmatized_sent_word_tag: {}".format(lemmatized_sent_word_tag))

		reshape_verb_dict = load_reshape_wordnet_dict_2('v')
		reshape_noun_dict = load_reshape_wordnet_dict_2('n')
		if question['difficulty'] == 'Hard':
			if question['sid'] in {**reshape_verb_dict, **reshape_noun_dict}:
				key_words, replace_dict = replace_sent_synonym_in_q(lemmatized_sent_word_tag, question)  #
				sub_record.append(replace_dict)
			else:
				sub_record.append(None)
		key_sent_words = lemmatize_vb_with_tag(normalized_sent_word_tag)
		print("key_words: {}".format(key_words))
		print("key_sent_words: {}".format(key_sent_words))

		for word in key_words:
			if word in key_sent_words:
				if word == 'the':
					score += 0.5
				elif word is single_vb:
					score += 1.5
				else:
					score += 1

		if score > max_match:
			matched_sentence = sentence
			max_match = score
			match_index = index

		index += 1

	if 'what' in q_start_word and 'say' in key_words:
		if Counter(matched_sentence)['\"'] == 1:
			matched_quote = matched_sentence
			for i in range(match_index + 1, len(sentences)):
				matched_quote += ' '
				matched_quote += sentences[i]
				if Counter(sentences[i])['\"'] > 0:
					return matched_quote.lower(), None

	print('sub_record: {}'.format(sub_record))
	if question['difficulty'] == 'Hard' and sub_record[match_index]:
		replace_dict = sub_record[match_index]
		update_qgraph(replace_dict, question)

		print('replace_dict: {}'.format(replace_dict))
	# print(match_index)
	# print(story_deps[match_index])
	return matched_sentence, story_deps[match_index]

def update_qgraph(replace_dict, question):
	l_question_after_sub = []
	qgraph = question['dep']
	n = len(qgraph.nodes)
	for i in range(1, n):
		q_graph_word = qgraph.nodes[i]['word'][:]
		if q_graph_word in replace_dict:
			qgraph.nodes[i]['word'] = replace_dict[q_graph_word]
			l_question_after_sub.append(replace_dict[q_graph_word])
		else:
			l_question_after_sub.append(q_graph_word)
	print('question after sub:')
	print(' '.join(l_question_after_sub))


def find_the_answer(matched_sentence, question):
	ps = PorterStemmer()
	string = []
	word_answer = tokenize_words(matched_sentence)
	word_question = tokenize_words(question)
	clue = word_question[0].lower()
	word_answer = nltk.pos_tag(word_answer)

	words = normalize_to_words(question)

	# define the main_verb (the verb appeared in question)
	main_verb = [w for (w, a) in words if a == 'v']
	# print(main_verb)
	# Get subject, no object
	if clue == 'what':
		# print(word_answer)
		for index, item in enumerate(word_answer):
			if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
				# find out sth is/are/are/was/were doing  OR sth is/are/are/was/were done,
				# then pick out the part before is/are/are/was/were as answer
				if 'VB' in word_answer[index - 1][1]:
					for item in word_answer[:index - 1]:
						# string = string+item[0]+" "
						string.append(item[0])
				# print("T1 Find the subject of -ing or -ed:", string)
				# find out sth/sb do sth, find out sth -ing -ed as the adjective
				# then pick out the part after the verb as answer
				else:
					for item in word_answer[index + 1:]:
						# string = string+item[0]+" "
						string.append(item[0])
				# print("T2 Find the object", string)
	if clue == 'why':
		found = False
		for item in tokenize_words(matched_sentence):
			if item == 'because':
				found = True
			if 'because' not in item and found == True:
				# string = string+item+" "
				string.append(item)
	# print("string: "+string)

	if clue == 'who':
		# print(word_answer)
		for index, item in enumerate(word_answer):
			if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
				# find out sth/sb do sth
				# then pick out the part before the verb as answer
				for item in word_answer[:index]:
					if 'VB' not in item[1]:
						# string = string+item[0]+" "
						string.append(item[0])
			# print("T3 Find the subject who type question", string)

	if len(string) == 0:
		return matched_sentence

	return " ".join(string)


"""
This function will cnt all nouns cnts in the story and 
save most popular n entities to global variable story_entities.
"""


def entity_counts(question, story, n):
	if story['sid'] in story_entities.keys():
		return

	cnt = Counter()
	sentences = None
	if 'Sch' in question['type']:
		sentences = get_sentences(story['sch'])
	else:
		sentences = get_sentences(story['text'])

	for sent in sentences:
		for w, t in sent:
			if "NN" in t:
				lemma_word = lemmatizer.lemmatize(w.lower(), 'n')
				if lemma_word not in cnt.keys():
					cnt[lemma_word] = 1
				else:
					cnt[lemma_word] += 1

	story_entities[story['sid']] = cnt.most_common(n)


def get_answer_with_overlap(question, story):
	answer = ""
	matched_sentence = ""
	matched_deps = None
	if 'Sch' in question['type']:
		matched_sentence, matched_deps = match_sent_from_q(question,
														   tokenize_sentences(
															   story['sch']),
														   story["sch_dep"])

	# answer = find_the_answer(matched_sentence,question['text'])
	else:
		matched_sentence, matched_deps = match_sent_from_q(question,
														   tokenize_sentences(
															   story['text']),
														   story["story_dep"])

	# answer = find_the_answer(matched_sentence,question['text'])
	answer = matched_sentence
	return answer, matched_sentence, matched_deps


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

	if q_start_word in ("when", "where", "why"):
		sentence_words = nltk.word_tokenize(matched_sentence)
		sentence_word_tag = nltk.pos_tag(sentence_words)
		answer_tree = chunk.find_candidates([sentence_word_tag], chunker,
											q_start_word)
		# if we found the answer
		if answer_tree:
			answer = " ".join([token[0] for token in answer_tree[0].leaves()])
		else:
			#print("Its raw_sent_answer:")
			answer = raw_sent_answer
	else:
		answer = raw_sent_answer
	return answer


def get_answer_with_deps(question, matched_deps, raw_sent_answer,
						 matched_sentence):
	answer = ""
	question_sents = get_sentences(question["text"])
	# start word of the question, ex: what, why, where, when, who
	q_start_word = question_sents[0][0][0].lower()
	qgraph = question["dep"]
	sgraph = matched_deps

	q_start_sets = set(
		["when", "where", "what", "why", "who", "how", "did", "had"])
	if verbose:
		q_start_sets = set([target_qstart])

	if q_start_word in q_start_sets:
		answer = dependency.find_answer(qgraph, sgraph, lemmatizer,
										q_start_word)
		# print(question["text"])
		# print(answer)
		# if we found the answer
		if not answer:
			answer = raw_sent_answer
	else:
		answer = raw_sent_answer
	return answer


def get_verbose(question, q_start_word, raw_sent_answer, answer, matched_deps):
	with open("tmp.txt", "a") as f:
		f.write(question["text"] + "\n")
		f.write("Raw answer: {0}\n".format(raw_sent_answer))
		f.write("Our answer: {0}\n".format(answer))
		# print(matched_deps)
		# q = []
		f.write("-" * 50 + "\n")
		f.write("Question dependency:\n")
		f.write("-" * 50 + "\n")
		for node in question["dep"].nodes.values():
			if node["word"] == None:
				continue
			f.write("address: {0}, word: {1}, rel: {2}, head: {3}\n".format(
				str(node["address"]), node["word"], node["rel"],
				str(node["head"])))
		# q.append((node["address"],node["word"], node["rel"], node["head"]))

		# l = []
		f.write("-" * 50 + "\n")
		f.write("Matched sentence dependency:\n")
		f.write("-" * 50 + "\n")
		if matched_deps != None:
			for node in matched_deps.nodes.values():
				if node["word"] == None:
					continue
				f.write("address:{0}, word:{1}, rel:{2}, head:{3}\n".format(
					str(node["address"]), node["word"], node["rel"],
					str(node["head"])))
			# l.append((node["address"],node["word"], node["rel"], node["head"]))
		else:
			f.write("Matched deps is None")
		# f.write("\n")
		f.write("*" * 50)
		f.write("\n")


def get_answer(question, story):
	print('=========================================')
	raw_sent_answer, matched_sentence, matched_deps = get_answer_with_overlap(
		question, story)

	return raw_sent_answer
	# print(normalize_and_lemmatize(question['text']))
	# if 'Sch' in question['type']:
	#   print(story['sch'])
	#   print(text_to_sent_words(story['sch']))
	# else:
	#   print(story['text'])
	#   print(text_to_sent_words(story['text']))
	# print(matched_sentence)

	# answer = raw_sent_answer
	question_sents = get_sentences(question["text"])
	q_start_word = question_sents[0][0][0].lower()
	q_start_list.append(q_start_word)

	if matched_deps != None:
		answer = get_answer_with_deps(question, matched_deps, raw_sent_answer,
									  matched_sentence)
	else:
		answer = raw_sent_answer

	if q_start_word=="where" and answer == raw_sent_answer:
		answer = get_answer_with_chunck(question, matched_sentence, raw_sent_answer)


	# answer = get_answer_with_chunck(question, matched_sentence, raw_sent_answer)
	# print("{Sentence}:", matched_sentence)
	# print("{Answer}:", answer)
	# print("\n")

	entity_counts(question, story, 2)
	if verbose and q_start_word == target_qstart:
		get_verbose(question, q_start_word, raw_sent_answer, answer,
					matched_deps)

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
	# if(question['type'] == "sch"):
	#    text = story['sch']
	###     Your Code Goes Here         ###

	# answer = "whatever you think the answer is"

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
	QA.run_score(set(['Hard']), set(['what', 'who']))


def main():
	# set evaluate to True/False depending on whether or
	# not you want to run your system on the evaluation
	# data. Evaluation data predictions will be saved
	# to hw6-eval-responses.tsv in the working directory.
	# if verbose:
	# 	if os.path.exists("tmp.txt"):
	# 		os.remove("tmp.txt")
	# run_qa(evaluate=False)
	# # # You can uncomment this next line to evaluate your
	# # # answers, or you can run score_answers.py
	# score_answers()
	#type_answer(target_qstart, q_start_list)
	run_qa_with_score()

# run_qa_with_score()

if __name__ == "__main__":
	main()