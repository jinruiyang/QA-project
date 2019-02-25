import pandas as pd
import nltk, argparse
import numpy as np

def score_all_answers(gold, pred):
    all_scores = {"p": [], "r": [], "f": []}
    for row in gold.itertuples():

        print("-"*40)
        print("\nSCORING {}\n".format(row.Index))

        golds = row.answer.lower().split("|")
        scores = {"p": [], "r": [], "f": []}
        for i, gold_answer in enumerate(golds):
            gold_words = set(nltk.word_tokenize(gold_answer))
            pred_answer = pred.loc[row.Index]
            pred_words = set(nltk.word_tokenize(pred_answer.answer.lower()))

            # true positives
            tp = len(gold_words - (gold_words - pred_words))

            # false positives
            fp = len(pred_words - gold_words)

            # false negatives
            fn = len(gold_words - pred_words)

            precision = tp / (tp + fp)*1.0
            recall = tp / (tp + fn)*1.0
            if recall + precision == 0:
                f1 = 0.0
            else:
                f1 = (2 * recall * precision) / (recall + precision)

            scores["f"].append(f1)
            scores["p"].append(precision)
            scores["r"].append(recall)

        best = np.argmax(scores["f"])
        best_gold = golds[best]

        print('Comparing Gold   "{}"\n      and Resp   "{}"'.format(best_gold, pred_answer.answer))
        all_scores["p"].append(scores["p"][best])
        all_scores["r"].append(scores["r"][best])
        all_scores["f"].append(scores["f"][best])

        print("\nRECALL:    {:.3f}\nPRECISION: {:.3f}\nF-measure: {:.3f}\n".format(
            scores["r"][best], scores["p"][best], scores["f"][best]))

    print("-" * 40)
    return np.mean(all_scores["p"]), np.mean(all_scores["r"]), np.mean(all_scores["f"])


def run_scoring(gold, pred):
    p, r, f = score_all_answers(gold, pred)

    print("\n\nFinished processing {} questions".format(gold.shape[0]))
    print("*************************************************************************\n")
    print("FINAL RESULTS\n\n")

    print("AVERAGE RECAL =     {:.4f}".format(r))
    print("AVERAGE PRECISION = {:.4f}".format(p))
    print("AVERAGE F-MEASURE = {:.4f}".format(f))
    print("\n*************************************************************************\n")


def main():
    import qa_engine.base as qa
    print("Computing QA Performance:")
    print("  * answer key:", qa.ANSWER_FILE)
    print("  * predictions file:", qa.RESPONSE_FILE)
    gold = pd.read_csv(qa.DATA_DIR + qa.ANSWER_FILE, index_col="qid", sep="\t")
    pred = pd.read_csv(qa.RESPONSE_FILE, index_col="qid", sep="\t")
    run_scoring(gold, pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assignment 6')
    parser.add_argument("answer_fname", help='Answer key file')
    parser.add_argument("response_fname", help='Your response file')
    args = parser.parse_args()

    gold = pd.read_csv(args.answer_fname, index_col="qid", sep="\t")
    pred = pd.read_csv(args.response_fname, index_col="qid", sep="\t")
    run_scoring(gold, pred)


