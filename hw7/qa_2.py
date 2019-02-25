
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk
import re, sys, operator
import dependency as dp

lemma = nltk.wordnet.WordNetLemmatizer()

def remove_stop_words(words, should_remove=False):
    if should_remove:
        stopwords = nltk.corpus.stopwords.words('english')
    else:
        stopwords = []
    filtered = [w.lower() for w in words if w not in stopwords]
    return filtered

def tokenize_sentences(paragraph):
    return [sentence for sentence in nltk.sent_tokenize(paragraph)]

# takes in a single sentence in string format
def tokenize_words(sentence):
    return [word for word in nltk.word_tokenize(sentence)]


"""
    Just get a sentence that is relative in the question and do for the
    matching by the sentence. The input for this is pos_tag of the question.
"""
def get_a_relative_question(question):
    words = tokenize_words(question)
    lowered = [w.lower() for w in words]

    filtered = []
    five_w = ['who', 'when', 'where', 'why', 'how', 'what', '?']

    for item in lowered:
        if item not in five_w:
            filtered.append(item)

    return filtered

def get_a_relative_graph(graph):
    words = dp.get_words_tag_from_graph(graph)
    filtered = []
    for (x,y) in words:
        if 'V' in y:
            filtered.append((x,y))
        if 'N' in y:
            filtered.append((x,y))

    return filtered

#def get_a_relative_question_in_deps(graph):

"""
    Find the main_noun and main_verb in the pos_tag list.
    Return the main_noun and main_verb in the given pos_tag.
"""
def find_key_words(words):
    lowered = [w.lower() for w in words]
    pos_tag = nltk.pos_tag(lowered)
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

def find_key_words_for_graph(words):
    #print(words)
    main_noun = []
    main_verb = []

    for (x,y) in words:
        if 'NN' in y:
            main_noun.append(x)
        elif 'V' in y:
            main_verb.append(x)
        else:
            main_noun.append(x)

    return main_noun, main_verb

"""
    The normalize_the_sentence gets an input of a sentence
    and returns a list in pos_tag (x,y) where x is a word and
    y is the part of speech of that word.

    But before doing so, it also uses stem and returns it.
"""
def stemming_the_question(words):
    #main_noun, main_verb = find_key_words(words)
    #main_noun = remove_stop_words(main_noun, True)
    main_noun, main_verb = find_key_words_for_graph(words)
    #rint(main_noun)
    #print(main_verb)
    final = []
    for item in main_verb:
        if 'ing' in item or 'ed' in item:
            word = lemma.lemmatize(item, 'v')
            final.append((word, 'v'))
        final.append((item, 'v'))

    for item in main_noun:
        final.append((item, 'n'))

    return final

def stemming_the_sentence(sentence):
    words = tokenize_words(sentence)
    pos_tag = nltk.pos_tag(words)

    # stopwords = nltk.corpus.stopwords.words('english')
    # stopwords.remove('from')

    final = []
    for (x,y) in pos_tag:
        if 'V' in y:
            if x == 'felt':
                word = 'feel'
            else:
                word = lemma.lemmatize(x, 'v')
        else:
            word = x

        final.append(word.lower())

    return final


"""
    This will compare the sentence with a question. We are going to
    find the main_verb in the question and also the main_noun in the question.
    Those two elements will be the key to find the sentence that answer the
    question.
"""


def compare_sentence(question, sentences, graph):
    max_match = 0
    matched_sentence = "couldn't find the answer"

    #rel_words = get_a_relative_question(question)
    rel_words = get_a_relative_graph(graph)
    #rel_words = stemming_the_question(rel_words)
    rel_words = stemming_the_question(rel_words)

    final_index = 0

    for sentence in sentences:
        count = 0
        rel_sentence = stemming_the_sentence(sentence)

        for (x,y) in rel_words:
            if x in rel_sentence:
                count += 1

        if count > max_match:
            matched_sentence = sentence
            final_index = sentences.index(sentence)
            max_match = count

    if matched_sentence == "couldn't find the answer":
        print('#################')
        print(question)
        print(rel_words)
        print(sentences)

    return final_index, matched_sentence


def get_answer(question, story):

    answer = "don't know the answer"
    index = None
    graph = question['dep']

    if 'Sch' in question['type']:
        index, answer = compare_sentence(question['text'], tokenize_sentences(story['sch']), graph)

    else:
        index, answer = compare_sentence(question['text'], tokenize_sentences(story['text']), graph)

    #return answer

    words = tokenize_words(question['text'])
    prev_sent = answer
    #graph = question['dep']
    #print(graph)
    story_graph = None

    """########################################
        Defines whether the type is sch or story
    ########################################"""
    if 'Sch' in question['type']:
        story_graph = story['sch_dep']
        #print('sch_dep')
    else:
        story_graph = story['story_dep']
        #print('text')

    if index >= len(story_graph):
        return answer

    dep_sentence = story_graph[index]


    """######################################
        If the question starts with 'who'
    ######################################
    """
    keyword = None

    if words[0] == 'Who' or words[0] == 'who':
        temp = dp.get_keyword_from_sentence(dep_sentence, 'who', " ")
        if temp:
            answer = temp
            return answer


    """######################################
        If the question starts with other than 'who'
    ######################################
    """

    if words[0] == 'What' or words[0] == 'what':
        main_word = dp.get_root(graph)
        main_word = dp.get_keyword_from_question(graph)
        keyword = dp.get_keyword_from_sentence(dep_sentence, 'What', main_word)

    if words[0] == 'Why' or words[0] == 'why':
        main_word = dp.get_root(graph)
        main_word = dp.get_keyword_from_question(graph)
        keyword = dp.get_keyword_from_sentence(dep_sentence, 'Why', main_word)


    if keyword:
        answer = keyword
        return answer

    #print('========graph==========')
    #print(question['text'])
    #print('=========main word=====')
    #print(main_word)
    #print('=========sentence=======')
    #print(prev_sent)
    #print(dep_sentence)
    #if not keyword:
    #    print(dep_sentence)
    print(answer)

    #print(dep_sentence)

    """
    #### returns answer ###
    """
    return answer





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

if __name__ == "__main__":
    main()
    #lemma = nltk.wordnet.WordNetLemmatizer()
    #print(lemma.lemmatize('felt','v'))
