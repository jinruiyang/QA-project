'''
Created on May 14, 2014
@author: Reid Swanson

Modified on May 21, 2015
'''

import re, sys, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from qa_engine.base import QABase
#import spacy
#spacy_parser = spacy.load('en_core_web_sm')


lmtzr = WordNetLemmatizer()

# Our simple grammar from class (and the book)
# Grammer set in fronter will be caught first, make sure no similar and simple ones put in fronter
GRAMMAR =   """
            N: {<PRP.*>|<NN.*>}
            Vb: {<V.*>+}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            CAUSE: {<PP> <Vb>}
            VP: {<TO>? <Vb> <DT>? (<NP>|<PP>)*}
            SUBJ_V_OBJ: {<NP> <VP> (<DT>? <IN>? <VP>)?}
            OBJ_V_SUBJ: {<NP> <Vb> <ADJ>? <VP>}
            
            """



LOC_PP = set(["in", "on", "at","under","behind","above","between","by","from","under","to"])
TIME_PP = set(["in", "on", "at", "since", "ago", "before", "by"]) # For a specific time, not range
CAUSE_PP = set(["because", "due", "to"])

def build_chunker():
    chunker = nltk.RegexpParser(GRAMMAR)
    return chunker

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences

def pp_filter(subtree):
    return subtree.label() == "PP"


def who_filter(subtree):
    return subtree.label() == "SUBJ_V_OBJ"

def what_filter(subtree):
    return subtree.label() in ("SUBJ_V_OBJ", "OBJ_V_SUBJ")

def cause_filter(subtree):
    return subtree.label() == "CAUSE"

# def when_filter(subtree):
#     return subtree.label() in 

def is_location(prep):
    return prep[0] in LOC_PP

def is_time(prep):
    return prep[0] in TIME_PP

def is_cause(prep):
    return prep[0] in CAUSE_PP

# for question asking which place, ex: "where"
def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # PP stands for prepositional phrases, ex: on, at, in........
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations
    
    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)
    return locations

# for question asking which animal or people, ex: "who"
def find_who(tree):
    who = []
    for subtree in tree.subtrees(filter=who_filter):
        who.append(subtree)
    return who

# for question asking which animal or people, ex: "who"
def find_what(tree):
    what = []
    for subtree in tree.subtrees(filter=what_filter):
        what.append(subtree)
    return what

def find_time(tree):
    time = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_time(subtree[0]):
            time.append(subtree)
    return time

def find_cause(tree):
    cause = []
    for subtree in tree.subtrees(cause_filter):
        if is_cause(subtree[0]):
            cause.append(subtree)
    return cause

# old version
def find_candidates_1(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = find_locations(tree)
        candidates.extend(locations)
        
    return candidates

def find_candidates(sentences, chunker,qstart):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        # print("This is the whole tree of:", sent)
        # print("The subtrees")
        if qstart == "where":
            locations = find_locations(tree)
            candidates.extend(locations)
        elif qstart == "who":
            who = find_who(tree)
            candidates.extend(who)
        elif qstart == "what":
            what = find_what(tree)
            candidates.extend(what)
        elif qstart == "when":
            time = find_time(tree)
            candidates.extend(time)
        elif qstart == "why":
            cause = find_cause(tree)
            candidates.extend(cause)
        else:
            print("*****Unseen start word*****")
        # print("Found subtrees:")
        # print("-"*50)

    # # Navigate parse tree with spacy (get dependency)
    # raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    # for sent in raw_sentences:
    #     doc = spacy_parser(sent)
    #     for chunk in doc.noun_chunks:
    #         print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)

    return candidates

def lemma_with_tags(sentences):
    # only lemma these tags
    lemma_sent = []+sentences
    lemma_tags = set("VB") 

    for i in range(len(lemma_sent)):
        for j in range(len(lemma_sent[i])):
            token = lemma_sent[i][j]
            if token[1][:2] in lemma_tags:
                lemma_sent[i][j][0] = lmtzr.lemmatize(lemma_sent[i][j][0])
    return lemma_sent

def find_sentences(patterns, sentences):
    # Get the raw text of each sentence to make it easier to search using regexes
    # raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    
    # Do lemmatize on reviews so we can match them with lemma keywords
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in lemma_with_tags(sentences)]

    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
                break # select the sentence if any word in pattern exits in the sent
        if matches:
            result.append(sent)
            
    return result


if __name__ == '__main__':
    # Our tools
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    
    question_id = "fables-01-1"

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]

    # Apply the standard NLP pipeline we've seen before
    sentences = get_sentences(text)
    
    # Assume we're given the keywords for now
    # What is happening
    verb = "sitting"
    # Who is doing it
    subj = "crow"
    # Where is it happening (what we want to know)
    loc = None
    
    # Might be useful to stem the words in case there isn't an extact
    # string match
    subj_stem = lmtzr.lemmatize(subj, "n")
    verb_stem = lmtzr.lemmatize(verb, "v")
    
    # Find the sentences that have all of our keywords in them
    # How could we make this better?
    crow_sentences = find_sentences([subj_stem, verb_stem], sentences)
    
    # Extract the candidate locations from these sentences
    locations = find_candidates_1(crow_sentences, chunker)
    
    # Print them out
    for loc in locations:
        print(loc)
        print(" ".join([token[0] for token in loc.leaves()]))
