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


def load_noun_ids(noun_csv="Wordnet_nouns.csv"):
    return load_wordnet_ids("{}/{}".format(DATA_DIR, noun_csv))


def load_verb_ids(verb_csv="Wordnet_verbs.csv"):
    return load_wordnet_ids("{}/{}".format(DATA_DIR, verb_csv))


# the reshape dict looks like: {story_id: {synset_id: word, synset_id: word,...}, story_id:{}...}
def reshape_wordnet_dict(wordnet_dict, type):
    reshape_dict = {}
    for synset_id, inner_dict in wordnet_dict.items():
        if type == 'v':
            word = inner_dict['story_verb']
        if type == 'n':
            word = inner_dict['story_noun']
        story_id_raw = inner_dict['stories'] # could have more than one story_id here
        l_story_id_str = story_id_raw.strip('{}').split(', ')
        for story_id_str in l_story_id_str:
            story_id = story_id_str.strip("\'").split('.')[0]
            if reshape_dict.get(story_id):
                story_dict = reshape_dict[story_id]
                story_dict[synset_id] = word
            else:
                story_dict = {synset_id: word}
                reshape_dict[story_id] = story_dict
    return reshape_dict


def load_reshape_wordnet_dict(type='v'):
    if type == 'v':
        return reshape_wordnet_dict(load_verb_ids(), type)
    if type == 'n':
        return reshape_wordnet_dict(load_noun_ids(), type)


if __name__ == "__main__":

    ## You can use either the .csv files or the .dict files.
    ## If you use the .dict files, you MUST use "rb"!

    noun_ids = load_noun_ids()
    verb_ids = load_verb_ids()
    print(noun_ids)
    print(verb_ids)
    reshape_noun_ids = reshape_wordnet_dict(noun_ids, 'n')
    reshape_verb_ids = reshape_wordnet_dict(verb_ids, 'v')
    print(reshape_noun_ids)
    print(reshape_verb_ids)
