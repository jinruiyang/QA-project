from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


stopwords = set(nltk.corpus.stopwords.words("english"))

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP> <POS>? <NP>?}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """
def find_locations(tree):           # 'where' question
    LOC_PP = set(["in", "on", "at", "under"])
    def is_location(prep):
        return prep[0] in LOC_PP

    def pp_filter(subtree):
        return subtree.label() == "PP"

    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)
    
    return locations

def find_subject(tree):              # 'who' question
    def np_filter(subtree):
        return subtree.label() == "NP"
    subject = []
    for subtree in tree.subtrees(filter=np_filter):
        # if subtree[0] in ['a','the']:
        subject.append(subtree)
    return subject

def find_object(tree):            # 'what' question
    def np_filter(subtree):
        return subtree.label() == "VP"
    object = []
    for subtree in tree.subtrees(filter=np_filter):
        # if subtree[0] in ['a','the']:
        object.append(subtree)
    return object

def find_time(tree):              # 'when' question
    time = []
    ...
    return time
def find_reason(tree):            #'why' questiopn
    reason = []
    ...
    return reason
def find_candidates(sentences, chunker, question_type):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        # print(tree)
        if question_type == 'where':
            locations = find_locations(tree)
            candidates.extend(locations)
        if question_type == 'when':
            pass
        if question_type == 'what':
            objects = find_object(tree)
            candidates.extend(objects)
        if question_type == 'who':
            subjects = find_subject(tree)
            candidates.extend(subjects)
        if question_type == 'why':
            pass
    return candidates