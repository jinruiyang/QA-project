
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import baseline as bs
import chunk as ck

stopwords = set(nltk.corpus.stopwords.words("english"))


def remove_stop(words):
    return [word.lower() for word in words if word.lower() not in stopwords]

def get_answer(question, story):
    lmtzr = WordNetLemmatizer()
    story_text = story["text"]
    sch = story["sch"]
    question_text = question['text']

    q_words = nltk.word_tokenize(question_text.lower())

    qst_bow = bs.get_bow(bs.get_sentences(question_text)[0])
    key_words = []
    for each in qst_bow:
        if not each == '?':
            # syns = wordnet.synsets(each)
            # for syn in syns:
            #     key_words.append(syn.lemmas()[0].name())
            #     key_words.append(syn.lemmas()[0].name().capitalize())
            key_words.append(each)
            key_words.append(lmtzr.lemmatize(each,pos='n'))
            key_words.append(lmtzr.lemmatize(each,pos='v'))
            key_words.append(each.capitalize())
            key_words.append(lmtzr.lemmatize(each,pos='n').capitalize())
            key_words.append(lmtzr.lemmatize(each,pos='v').capitalize())
    key_words = set(key_words)
    # print(key_words)
    story_sents = bs.get_sentences(story_text)
    #sch_sents = bs.get_sentences(sch)
    answer, sec_answer = bs.baseline(key_words, story_sents)
    #sch_answer = bs.baseline(key_words, sch_sents)
    answer = [" ".join(t[0] for t in answer)]
    sec_answer = [" ".join(t[0] for t in sec_answer)]
    #sch_answer = [" ".join(t[0] for t in sch_answer)]
    located_sentence = answer[0] + " " + sec_answer[0]
    #located_sentence = answer[0] + " " + sch_answer[0]

    tagged_located_sentence = bs.get_sentences(located_sentence)
    chunker = nltk.RegexpParser(ck.GRAMMAR)

    #print(tagged_located_sentence)

    if 'where' in q_words:
        locations = ck.find_candidates(tagged_located_sentence, chunker, 'where')
        answer = ""
        found = False
        for loc in locations:
            found = True
            # print(loc)
            for each in loc.leaves():
                answer += each[0]+' '
        if found:
            return answer.lower()
        else:
            return located_sentence
    elif 'what' in q_words:
        object = ck.find_candidates(tagged_located_sentence, chunker, 'what')
        answer = ""
        found = False
        for person in object:
            found = True
            # print(loc)
            '''
            for each in person.leaves():
                if 'N' in each[1]:
                  #print(each)
                  answer += each[0] + ' '
            #answer += ','
            '''
            for each in person.leaves():
                answer += each[0]+' '
            answer += ','
        if found:
            return answer.lower()
        else:
            return located_sentence
    elif 'who' in q_words:
        people = ck.find_candidates(tagged_located_sentence, chunker, 'who')
        answer = ""
        found = False
        for person in people:
            found = True
            # print(loc)
            for each in person.leaves():
                answer += each[0]+' '
            answer += ','
        if found:
            return answer.lower()
        else:
            return located_sentence
    elif 'when' in q_words:
        ...
        return located_sentence
    elif 'why' in q_words:
        ...
        return located_sentence
    else:
        return located_sentence




#############################################################
###     Dont change the code in this section
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

#############################################################


def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
