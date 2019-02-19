'''
Created on May 14, 2014
@author: Reid Swanson

Modified on May 21, 2015
'''

import re, sys, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from qa_engine.base import QABase




# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at"])


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences

def pp_filter(subtree):
    return subtree.label() == "PP"

def is_location(prep):
    return prep[0] in LOC_PP

def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
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

def find_candidates(sentences, chunker):
    candidates = []
    for sent in crow_sentences:
        tree = chunker.parse(sent)
        # print(tree)
        locations = find_locations(tree)
        candidates.extend(locations)
        
    return candidates

def find_sentences(patterns, sentences):
    # Get the raw text of each sentence to make it easier to search using regexes
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    
    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
                break
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
    locations = find_candidates(crow_sentences, chunker)
    
    # Print them out
    for loc in locations:
        print(loc)
        print(" ".join([token[0] for token in loc.leaves()]))
