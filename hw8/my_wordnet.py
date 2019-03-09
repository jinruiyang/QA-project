import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn

DATA_DIR = "./wordnet"

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

#noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
#verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

def look_for_noun_in_dataset(synsets):
    result = None
    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    for synset_id, items in noun_ids.items():
        noun = items['story_noun']
        if synset_id == synsets:
            wordnet.append(noun)

    return result


def look_for_verb_in_dataset(synsets):
    result = None
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

    for synset_id, items in verb_ids.items():
        verb = items['story_verb']
        if synset_id == synsets:
            wordnet.append(verb)

    return result

"""
    Look for hyponyms, hypernym, synonyms, etc..
"""
def look_for_everything(synsets):
    all_synsets = []
    for synset in synsets:
        #hyponyms
        hypo = synset.hyponyms()
        for item in hypo:
            all_synsets.append(item)

        #hypernym
        hyper = synset.hypernyms()
        for item in hyper:
            all_synsets.append(item)

    #print(result)
    return all_synsets

#if __name__ == "__main__":
#    ro_synset = wn.synsets('rodent')
#    look_for_everything(ro_synset)

"""
def main(word, pos):
    wordnet = None
    if pos == 'n':
        synsets = look_for_noun_in_dataset(word)
        if not synset:
            synsets = wn.synsets(word)
    else:
        synsets = look_for_verb_in_dataset(word)
        if not synset:
            synsets = wn.synsets(word)

    return look_for_everything(synsets)
"""

