'''
import pandas as pd
from nltk.parse import DependencyGraph
from nltk.tree import Tree
from collections import defaultdict

HW = 7
DATA_DIR = "data/"
QUESTION_FILE = "hw{}-questions.tsv".format(HW)
ANSWER_FILE = "hw{}-answers.tsv".format(HW)
STORIES_FILE = "hw{}-stories.tsv".format(HW)

RESPONSE_FILE = "hw{}-responses.tsv".format(HW)
EVAL_RESPONSE_FILE = "hw{}-eval-responses.tsv".format(HW)

EVAL_QUESTIONS = "hw{}-eval-questions.tsv".format(HW)
EVAL_STORIES = "hw{}-eval-stories.tsv".format(HW)


import math

from collections import defaultdict


def prepare_deps(raw_deps):

    if isinstance(raw_deps, float) and math.isnan(raw_deps):
        return []
    return [DependencyGraph(dep, top_relation_label="root") for dep in raw_deps.split("\n\n")
            if len(dep) > 2]


def prepare_pars(raw_pars):
    if isinstance(raw_pars, float) and math.isnan(raw_pars):
        return []

    return [Tree.fromstring(line.strip().rstrip(" \n\t"))
            for line in raw_pars.split("\n") if len(line) > 2]


def prepare_story_data(df):
    stories = {}
    for row in df.itertuples():
        this_story = {
            "story_dep": prepare_deps(row.story_dep),
            "sch_dep": prepare_deps(row.sch_dep),
            "sch_par": prepare_pars(row.sch_par),
            "story_par": prepare_pars(row.story_par),
            "sch": row.sch,
            "text": row.text,
            "sid": row.sid,
        }
        stories[row.sid] = this_story
    return stories


def prepare_questions(df):
    questions = {}
    for row in df.itertuples():
        this_qstn = {
            "dep": prepare_deps(row.dep)[0],
            "par": prepare_pars(row.par)[0],
            "text": row.text,
            "sid": row.sid,
            "difficulty": row.difficulty,
            "type": row.type,
            "qid": row.qid
        }
        questions[row.qid] = this_qstn
    return questions


class QABase(object):

    def __init__(self, evaluate=False):
        self.evaluate = evaluate

        if evaluate:
            qstn_file = EVAL_QUESTIONS
            story_file = EVAL_STORIES
        else:
            qstn_file = QUESTION_FILE
            story_file = STORIES_FILE

        self._stories = prepare_story_data(pd.read_csv(DATA_DIR + story_file, sep="\t"))
        self._questions = prepare_questions(pd.read_csv(DATA_DIR + qstn_file, sep="\t"))
        self._answers = {q["qid"]: "" for q in self._questions.values()}


    @staticmethod
    def answer_question(question, story):
        raise NotImplemented


    def get_question(self, qid):
        return self._questions.get(qid)


    def get_story(self, sid):
        return self._stories.get(sid)


    def run(self):
        for qid, q in self._questions.items():
            a = self.answer_question(q, self._stories.get(q["sid"]))
            self._answers[qid] = {"answer": a, "qid": qid}


    def save_answers(self, fname=None):
        if fname is None:
            if self.evaluate:
                fname = EVAL_RESPONSE_FILE
            else:
                fname = RESPONSE_FILE
        df = pd.DataFrame([a for a in self._answers.values()])
        df.to_csv(fname, sep="\t", index=False)
'''

import pandas as pd
from nltk.parse import DependencyGraph
from nltk.tree import Tree
from collections import defaultdict
import nltk
import numpy as np

HW = 7
DATA_DIR = "data/"
QUESTION_FILE = "hw{}-questions.tsv".format(HW)
ANSWER_FILE = "hw{}-answers.tsv".format(HW)
STORIES_FILE = "hw{}-stories.tsv".format(HW)

RESPONSE_FILE = "hw{}-responses.tsv".format(HW)
EVAL_RESPONSE_FILE = "hw{}-eval-responses.tsv".format(HW)

EVAL_QUESTIONS = "hw{}-eval-questions.tsv".format(HW)
EVAL_STORIES = "hw{}-eval-stories.tsv".format(HW)


import math

from collections import defaultdict


def prepare_deps(raw_deps):

    if isinstance(raw_deps, float) and math.isnan(raw_deps):
        return []
    return [DependencyGraph(dep, top_relation_label="root") for dep in raw_deps.split("\n\n")
            if len(dep) > 2]


def prepare_pars(raw_pars):
    if isinstance(raw_pars, float) and math.isnan(raw_pars):
        return []

    return [Tree.fromstring(line.strip().rstrip(" \n\t"))
            for line in raw_pars.split("\n") if len(line) > 2]


def prepare_story_data(df):
    stories = {}
    for row in df.itertuples():
        this_story = {
            "story_dep": prepare_deps(row.story_dep),
            "sch_dep": prepare_deps(row.sch_dep),
            "sch_par": prepare_pars(row.sch_par),
            "story_par": prepare_pars(row.story_par),
            "sch": row.sch,
            "text": row.text,
            "sid": row.sid,
        }
        stories[row.sid] = this_story
    return stories


def prepare_questions(df):
    questions = {}
    for row in df.itertuples():
        this_qstn = {
            "dep": prepare_deps(row.dep)[0],
            "par": prepare_pars(row.par)[0],
            "text": row.text,
            "sid": row.sid,
            "difficulty": row.difficulty,
            "type": row.type,
            "qid": row.qid
        }
        questions[row.qid] = this_qstn
    return questions


class QABase(object):

    def __init__(self, evaluate=False):
        self.evaluate = evaluate

        if evaluate:
            qstn_file = EVAL_QUESTIONS
            story_file = EVAL_STORIES
        else:
            qstn_file = QUESTION_FILE
            story_file = STORIES_FILE

        self._stories = prepare_story_data(pd.read_csv(DATA_DIR + story_file, sep="\t"))
        self._questions = prepare_questions(pd.read_csv(DATA_DIR + qstn_file, sep="\t"))
        self._answers = {q["qid"]: "" for q in self._questions.values()}

    @staticmethod
    def answer_question(question, story):
        raise NotImplemented

    def get_question(self, qid):
        return self._questions.get(qid)


    def get_story(self, sid):
        return self._stories.get(sid)

    def run(self):
        for qid, q in self._questions.items():
            a = self.answer_question(q, self._stories.get(q["sid"]))
            self._answers[qid] = {"answer": a, "qid": qid}

    def save_answers(self, fname=None):
        if fname is None:
            if self.evaluate:
                fname = EVAL_RESPONSE_FILE
            else:
                fname = RESPONSE_FILE
        df = pd.DataFrame([a for a in self._answers.values()])
        df.to_csv(fname, sep="\t", index=False)

 # ----------------------new functions below defined by Howard-----------------

    def get_q_type(self, q):
        return 'sch' if 'Sch' in q['type'] else 'text'

    def get_q_startword(self, q_str):
        return nltk.word_tokenize(q_str)[0].lower()

    def print_text(self, text):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            print(sentence)

    def check_and_print_text(self, visited_sid_type, q):
        sid = q['sid']
        q_type = self.get_q_type(q)
        if (sid, q_type) not in visited_sid_type:
            story = self._stories.get(sid)
            print('=========================================')
            print(q_type)
            self.print_text(story[q_type])
            # print(story[q_type])
            visited_sid_type.add((sid, q_type))

    def calculate_and_print_scores(self, row, pred_str, all_scores):
        print("-" * 40)
        print("SCORING {}".format(row.Index))
        golds = row.answer.lower().split("|")
        scores = {"p": [], "r": [], "f": []}
        for i, gold_answer in enumerate(golds):
            gold_words = set(nltk.word_tokenize(gold_answer))
            pred_words = set(nltk.word_tokenize(pred_str.lower()))

            # true positives
            tp = len(gold_words - (gold_words - pred_words))

            # false positives
            fp = len(pred_words - gold_words)

            # false negatives
            fn = len(gold_words - pred_words)

            precision = tp / (tp + fp) * 1.0
            recall = tp / (tp + fn) * 1.0
            if recall + precision == 0:
                f1 = 0.0
            else:
                f1 = (2 * recall * precision) / (recall + precision)

            scores["f"].append(f1)
            scores["p"].append(precision)
            scores["r"].append(recall)

        best = np.argmax(scores["f"])
        best_gold = golds[best]

        print('Comparing Gold   "{}"\n      and Resp   "{}"'.format(best_gold,
                                                                    pred_str))
        all_scores["p"].append(scores["p"][best])
        all_scores["r"].append(scores["r"][best])
        all_scores["f"].append(scores["f"][best])

        print("\nRECALL:    {:.3f}\nPRECISION: {:.3f}\nF-measure: {:.3f}\n".format(
            scores["r"][best], scores["p"][best], scores["f"][best]))

    def print_final_score(self, all_scores, gold):
        p, r, f = np.mean(all_scores["p"]), np.mean(all_scores["r"]), np.mean(
            all_scores["f"])

        print("\n\nFinished processing {} questions".format(gold.shape[0]))
        print("*************************************************************************\n")
        print("FINAL RESULTS\n\n")

        print("AVERAGE RECAL =     {:.4f}".format(r))
        print("AVERAGE PRECISION = {:.4f}".format(p))
        print("AVERAGE F-MEASURE = {:.4f}".format(f))
        print("\n*************************************************************************\n")

    def run_score(self, q_startwords=set(['what', 'when', 'where', 'who', 'why', 'how', 'did', 'had'])):
        print(q_startwords)
        visited_sid_type = set()
        all_scores = {"p": [], "r": [], "f": []}
        gold = pd.read_csv(DATA_DIR + ANSWER_FILE, index_col="qid", sep="\t")
        for (qid, q), row in zip(self._questions.items(), gold.itertuples()):
            q_startword = self.get_q_startword(q['text'])
            if q_startword not in q_startwords:
                continue
            self.check_and_print_text(visited_sid_type, q)
            pred_str = self.answer_question(q, self._stories.get(q["sid"]))
            self._answers[qid] = {"answer": pred_str, "qid": qid}
            print('answer_type:' + q['type'])
            self.calculate_and_print_scores(row, pred_str, all_scores)
        self.print_final_score(all_scores, gold)





