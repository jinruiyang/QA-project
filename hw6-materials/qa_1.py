
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem import PorterStemmer
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
import nltk

lemma = nltk.wordnet.WordNetLemmatizer()

'''
def convert_tree_to_tag(tree):
    for subtree in tree.subtrees():
        print(subtree)
    print('--------------')
'''

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

    filtered = [(x,y) for (x,y) in pos_tag if x not in stopwords]

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
    five_w = ['who', 'when', 'where', 'why', 'how', 'what']

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

def compare_sentence(question, sentences):
    max_match = 0
    matched_sentence = "couldn't find the answer"

    rel_words = get_a_relative_question(question)
    rel_words = stemming_the_question(rel_words)

    print("question: "+question)
    #print("filtered: "+rel_string)
    #print(rel_words)
    for sentence in sentences:
        count = 0
        rel_sentence = stemming_the_sentence(sentence)
        #print(rel_sentence)
        for (x,y) in rel_words:
            if x in rel_sentence:
                count += 1

        if count > max_match:
            matched_sentence = sentence
            max_match = count

    #print(final_sentence)
    #print("answer: "+matched_sentence)
    return matched_sentence


def find_the_answer(matched_sentence,question):
    ps = PorterStemmer()
    string = []
    word_answer = tokenize_words(matched_sentence)
    word_question = tokenize_words(question)
    clue = word_question[0].lower()
    word_answer = nltk.pos_tag(word_answer)

    words = stemming_the_question(word_question)

    #define the main_verb
    main_verb = [w for (w,a) in words if a == 'v']
    #define the main noun
    main_noun = [w for(w,a) in words if a == 'n']
    print("main verb:", main_verb)
    #print("main noun:", main_noun)
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
                    print("T1 Find the subject of -ing or -ed:", string)
                # find out sth/sb do sth, find out sth -ing -ed as the adjective
                # then pick out the part after the verb as answer
                else:
                    for item in word_answer[index+1:]:
                        #string = string+item[0]+" "
                        string.append(item[0])
                    print("T2 Find the object", string)
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
                print("T3 Find the subject who type question", string)

    if len(string) == 0:
        return matched_sentence

    return " ".join(string)


def get_answer(question, story):
    answer = "couldn't find the answer"
    print('=========================================')
    if 'Sch' in question['type']:
        #convert_tree_to_tag(question['par'])
        matched_sentence = compare_sentence(question['text'], tokenize_sentences(story['sch']))
        print("before: "+matched_sentence)

        answer = find_the_answer(matched_sentence,question['text'])
        print("after: "+answer)
    else:
        matched_sentence = compare_sentence(question['text'], tokenize_sentences(story['text']))
        print("before: "+matched_sentence)
        answer = find_the_answer(matched_sentence,question['text'])
        print("after: "+answer)



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
