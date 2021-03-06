
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from qa_engine.type_answer import main as type_answer
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk
import dependency
#import preprocess
#import chunk

lemma = nltk.wordnet.WordNetLemmatizer()
q_start_list = []
# chunker = chunk.build_chunker()

# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences  

def convert_tree_to_tag(tree):
    for subtree in tree.subtrees():
        print(subtree)
    print('--------------')

def remove_stop_words(words, should_remove=False):
    if should_remove:
        stopwords = nltk.corpus.stopwords.words('english')
    else:
        stopwords = []
    filtered = [w.lower() for w in words if w not in stopwords]
    # print('before:', words)
    # print('after:', filtered)
    return filtered

def remove_question_words(words):
    lowered = [w.lower() for w in words]
    filtered = []
    question_words = ['?', "\'"]

    for word in lowered:
        if word not in question_words:
            filtered.append(word)
    return filtered

def tokenize_sentences(paragraph):
    return [sentence for sentence in nltk.sent_tokenize(paragraph)]

# takes in a single sentence in string format
def tokenize_words(sentence):
    return [word for word in nltk.word_tokenize(sentence)]


"""
    Find the main_noun and main_verb in the pos_tag list.
    Return the main_noun and main_verb in the given pos_tag.
"""
def find_key_words(words):
    lowered = [w.lower() for w in words]
    pos_tag = nltk.pos_tag(lowered)
    #print(pos_tag)
    main_noun = []
    main_verb = []
    for item in pos_tag:
        if 'NN' in item[1]:
            main_noun.append(item[0])
        elif 'VB' in item[1]:
            main_verb.append(item[0])
        else:
            main_noun.append(item[0])

    return main_noun, main_verb

"""
    The normalize_the_sentence gets an input of a sentence
    and returns a list in pos_tag (x,y) where x is a word and
    y is the part of speech of that word.

    But before doing so, it also uses stem and returns it.
"""
def stemming_the_question(words):
    ps = PorterStemmer()
    main_noun, main_verb = find_key_words(words)

    noun_filtered = remove_stop_words(main_noun)
    verb_filtered = remove_stop_words(main_verb)

    final = []
    for item in verb_filtered:
        if 'ing' in item or 'ed' in item:
            word = lemma.lemmatize(item, 'v')
            final.append((word, 'v'))
        else:
            final.append((item, 'v'))

    for item in noun_filtered:
        final.append((item, 'n'))

    return final

def stemming_the_sentence(sentence):
    ps = PorterStemmer()
    words = tokenize_words(sentence)
    pos_tag = nltk.pos_tag(words)

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('from')

    #filtered = [(x,y) for (x,y) in pos_tag if x not in stopwords]
    filtered = pos_tag
    #lemma = nltk.wordnet.WordNetLemmatizer()

    final = []
    for (x,y) in filtered:
        if 'VB' in y:
            if x == 'felt':
                word = 'feel'
            else:
                word = lemma.lemmatize(x, 'v')
        else:
            word = x

        #word = lemma.lemmatize(x)
        final.append(word.lower())

    return final


"""
    Just get a sentence that is relative in the question and do for the
    matching by the sentence. The input for this is pos_tag of the question.
"""
def get_a_relative_question(question):
    words = tokenize_words(question)
    lowered = [w.lower() for w in words]
    #print(lowered)
    #pos_tag = pos_tagger(question)
    filtered = []
    five_w = ['who', 'when', 'where', 'why', 'how', 'what', 'did', 'had', 'which']

    for item in lowered:
        if item not in five_w:
            filtered.append(item)

    return filtered


"""
    This will compare the sentence with a question. We are going to
    find the main_verb in the question and also the main_noun in the question.
    Those two elements will be the key to find the sentence that answer the
    question.
"""

def compare_sentence(question, sentences, story_deps):
    max_match = 0
    idx_match = 0
    matched_sentence = "couldn't find the answer"

    rel_words = get_a_relative_question(question)
    rel_words = stemming_the_question(rel_words)

    ############################################
    # TODO: use dep get keywords
    ############################################

    #print("question: "+question)
    #print("filtered: "+rel_string)
    #print(rel_words)
    for i in range(len(sentences)):
        sentence = sentences[i]
        count = 0
        rel_sentence = stemming_the_sentence(sentence)
        #print(rel_sentence)
        for (x,y) in rel_words:
            if x in rel_sentence:
                count += 1

        if count > max_match:
            matched_sentence = sentence
            max_match = count
            idx_match = i

    #print(final_sentence)
    #print("answer: "+matched_sentence)
    return matched_sentence, story_deps[idx_match]

def sch_compare_sentence(question, sentences):
    max_match = 0
    matched_sentence = "couldn't find the answer"

    rel_words = get_a_relative_question(question)
    rel_words = stemming_the_question(rel_words)

    possible_answer = []

    print("question: "+question)

    for sentence in sentences:
        count = 0
        rel_sentence = stemming_the_sentence(sentence)

        for (x,y) in rel_words:
            if x in rel_sentence:
                count += 1

        if count > max_match:
            possible_answer = []
            possible_answer.append(sentence)
            max_match = count

        if count == max_match:
            possible_answer.append(sentence)

    return matched_sentence


def find_the_answer(matched_sentence,question):
    ps = PorterStemmer()
    string = []
    word_answer = tokenize_words(matched_sentence)
    word_question = tokenize_words(question)
    clue = word_question[0].lower()
    word_answer = nltk.pos_tag(word_answer)

    words = stemming_the_question(word_question)

    #define the main_verb (the verb appeared in question)
    main_verb = [w for (w,a) in words if a == 'v']
    #print(main_verb)
    # Get subject, no object
    # if clue == 'what':
    #     for index, item in enumerate(word_answer):
    #         if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
    #             if 'VB' in word_answer[index-1][1]:
    #                 for item in word_answer[:index-1]:
    #                     #string = string+item[0]+" "
    #                     string.append(item[0])

    #             else:
    #                 for item in word_answer[index+1:]:
    #                     #string = string+item[0]+" "
    #                     string.append(item[0])

    # if clue == 'why':
    #     found = False
    #     for item in tokenize_words(matched_sentence):
    #         if item == 'because':
    #             found = True
    #         if 'because' not in item and found == True:
    #             #string = string+item+" "
    #             string.append(item)
    # #print("string: "+string)

    # if clue == 'who':
    #     for index, item in enumerate(word_answer):
    #         if 'VB' in item[1] and ps.stem(item[0]) in main_verb:
    #             for item in word_answer[:index]:
    #                 if 'VB' not in item[1]:
    #                     #string = string+item[0]+" "
    #                     string.append(item[0])

    if len(string) == 0:
        return matched_sentence

    return " ".join(string)


def get_answer_with_overlap(question, story):
    answer = ""
    matched_sentence = ""
    matched_deps = None
    if 'Sch' in question['type']:
        #convert_tree_to_tag(question['par'])
        matched_sentence, matched_deps = compare_sentence(question['text'], tokenize_sentences(story['sch']), story["sch_dep"])
        #print("before: "+matched_sentence)

        answer = find_the_answer(matched_sentence,question['text'])
        #print("after: "+answer)
    else:
        matched_sentence, matched_deps = compare_sentence(question['text'], tokenize_sentences(story['text']), story["story_dep"])

        #print("before: "+matched_sentence)
        answer = find_the_answer(matched_sentence,question['text'])
        #print("after: "+answer)

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

    if q_start_word in ("where","what"):
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
    
def get_answer_with_deps(question, matched_deps, raw_sent_answer, matched_sentence):
    
    answer = ""
    question_sents = get_sentences(question["text"])
    # start word of the question, ex: what, why, where, when, who
    q_start_word = question_sents[0][0][0].lower() 
    qgraph = question["dep"]
    sgraph = matched_deps

    if q_start_word in ("when"):

        answer = dependency.find_answer(qgraph ,sgraph, lemma, q_start_word)
        #print(question["text"])
        #print(answer)
        # if we found the answer
        if not answer:
            answer = raw_sent_answer
    else:
        answer = raw_sent_answer
    return answer

def get_answer(question, story):
    answer = "couldn't find the answer"
    raw_answer, matched_sentence, matched_deps = get_answer_with_overlap(question,story)

    
    
    
    question_sents = get_sentences(question["text"])
    # start word of the question, ex: what, why, where, when, who
    q_start_word = question_sents[0][0][0].lower()
    q_start_list.append(q_start_word)
    answer = get_answer_with_deps(question, matched_deps, raw_answer, matched_sentence)
    #answer = raw_answer
    

    # if q_start_word == "when":
    #     print(question["text"])
    #     print(raw_answer)
    #     print(answer)
    #     # print(matched_deps)
    #     l = []
    #     for node in matched_deps.nodes.values():
    #         l.append((node["address"],node["word"], node["rel"], node["head"]))
    #     print(l)
    #     print("-")
    




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
###     Dont change the code below here
#############################################################

class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()
    #type_answer("when",q_start_list)

if __name__ == "__main__":
    main()
    #lemma = nltk.wordnet.WordNetLemmatizer()
    #print(lemma.lemmatize('felt','v'))
    """
    synonyms = []
    antonyms = []
    for syn in wn.synsets("foolish"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print(synonyms)
    print(antonyms)
    """
