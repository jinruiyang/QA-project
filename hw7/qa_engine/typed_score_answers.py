import pandas as pd
import nltk, argparse
import numpy as np

def score_all_answers(gold, pred):
    # Updated to have separate scoring
    separate_scores = {"easy": {"p": [], "r": [], "f": []},
                       "medium": {"p": [], "r": [], "f": []},
                       "hard": {"p": [], "r": [], "f": []},
                       "discourse": {"p": [], "r": [], "f": []}}
    all_scores = {"p": [], "r": [], "f": []}

    for row in gold.itertuples():
        difficulty = row.difficulty.lower()

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

            # Updated to handle 0 divisor
            if len(pred_words) == 0:
                precision = 1.0
            elif tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp) * 1.0

            # Updated to handle 0 divisor
            if tp + fn == 0:
                recall = 0.0
            else:
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

        print("Difficulty:", difficulty)
        if difficulty.lower() not in separate_scores:
            print("unexpected difficulty:", difficulty)
            exit(1)
        else:
            separate_scores[difficulty]["p"].append(scores["p"][best])
            separate_scores[difficulty]["r"].append(scores["r"][best])
            separate_scores[difficulty]["f"].append(scores["f"][best])


        r, p, f = scores["r"][best], scores["p"][best], scores["f"][best]
        #print("===> ", r, p, f)
        print("\nRECALL:    {:.3f}\nPRECISION: {:.3f}\nF-measure: {:.3f}\n".format(r, p, f))

    print("-" * 40)
    print("done! \n")

    avg_easy_scores = (np.mean(separate_scores["easy"]["p"]), np.mean(separate_scores["easy"]["r"]), np.mean(separate_scores["easy"]["f"]))
    avg_medium_scores = (np.mean(separate_scores["medium"]["p"]), np.mean(separate_scores["medium"]["r"]), np.mean(separate_scores["medium"]["f"]))
    avg_hard_scores = (np.mean(separate_scores["hard"]["p"]), np.mean(separate_scores["hard"]["r"]), np.mean(separate_scores["hard"]["f"]))
    avg_discourse_scores = (np.mean(separate_scores["discourse"]["p"]), np.mean(separate_scores["discourse"]["r"]), np.mean(separate_scores["discourse"]["f"]))
    avg_all_scores = (np.mean(all_scores["p"]), np.mean(all_scores["r"]), np.mean(all_scores["f"]))
    return avg_easy_scores, avg_medium_scores, avg_hard_scores, avg_discourse_scores, avg_all_scores




def main():
    import qa_engine.base as qa
    #print(qa.ANSWER_FILE)
    #print(qa.RESPONSE_FILE)
    gold = pd.read_csv(qa.DATA_DIR + qa.ANSWER_FILE, index_col="qid")
    # Updated to handle empty string
    pred = pd.read_csv(qa.RESPONSE_FILE, index_col="qid", na_values=" ", keep_default_na=False)

    avg_easy_score, avg_medium_score, avg_hard_score, avg_discourse_score, avg_all_score = score_all_answers(gold, pred)

    print("\n\nFinished processing {} questions".format(gold.shape[0]))
    print("*************************************************************************\n")
    print("FINAL RESULTS\n\n")

    print("\n**************************EASY*****************************************\n")
    print("AVERAGE EASY RECAL =     {:.4f}".format(avg_easy_score[1]))
    print("AVERAGE EASY PRECISION = {:.4f}".format(avg_easy_score[0]))
    print("AVERAGE EASY F-MEASURE = {:.4f}".format(avg_easy_score[2]))

    print("\n**************************MEDIUM****************************************\n")
    print("AVERAGE MEDIUM RECAL =     {:.4f}".format(avg_medium_score[1]))
    print("AVERAGE MEDIUM PRECISION = {:.4f}".format(avg_medium_score[0]))
    print("AVERAGE MEDIUM F-MEASURE = {:.4f}".format(avg_medium_score[2]))

    print("\n*************************HARD*******************************************\n")
    print("AVERAGE HARD RECAL =     {:.4f}".format(avg_hard_score[1]))
    print("AVERAGE HARD PRECISION = {:.4f}".format(avg_hard_score[0]))
    print("AVERAGE HARD F-MEASURE = {:.4f}".format(avg_hard_score[2]))

    print("\n*************************DISCOURSE***************************************\n")
    print("AVERAGE DISCOURSE RECAL =     {:.4f}".format(avg_discourse_score[1]))
    print("AVERAGE DISCOURSE PRECISION = {:.4f}".format(avg_discourse_score[0]))
    print("AVERAGE DISCOURSE F-MEASURE = {:.4f}".format(avg_discourse_score[2]))


    print("\n*************************OVERALL*****************************************\n")
    print("AVERAGE RECAL =     {:.4f}".format(avg_all_score[1]))
    print("AVERAGE PRECISION = {:.4f}".format(avg_all_score[0]))
    print("AVERAGE F-MEASURE = {:.4f}".format(avg_all_score[2]))
    print("\n*************************************************************************\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assignment 8')
    parser.add_argument('-a', dest="answer_fname", help='Answer key file')

    parser.add_argument('-r', dest="response_fname", help='Your response file')
    args = parser.parse_args()

    gold = pd.read_csv(args.answer_fname, index_col="qid")
    # Updated to handle empty string
    pred = pd.read_csv(args.response_fname, index_col="qid", na_values=" ", keep_default_na=False)


    avg_easy_score, avg_medium_score, avg_hard_score, avg_discourse_score, avg_all_score = score_all_answers(gold, pred)
    print("\n\nFinished processing {} questions".format(gold.shape[0]))
    print("*************************************************************************\n")
    print("FINAL RESULTS\n\n")

    print("\n**************************EASY*****************************************\n")
    print("AVERAGE EASY RECAL =     {:.4f}".format(avg_easy_score[1]))
    print("AVERAGE EASY PRECISION = {:.4f}".format(avg_easy_score[0]))
    print("AVERAGE EASY F-MEASURE = {:.4f}".format(avg_easy_score[2]))

    print("\n**************************MEDIUM****************************************\n")
    print("AVERAGE MEDIUM RECAL =     {:.4f}".format(avg_medium_score[1]))
    print("AVERAGE MEDIUM PRECISION = {:.4f}".format(avg_medium_score[0]))
    print("AVERAGE MEDIUM F-MEASURE = {:.4f}".format(avg_medium_score[2]))

    print("\n*************************HARD*******************************************\n")
    print("AVERAGE HARD RECAL =     {:.4f}".format(avg_hard_score[1]))
    print("AVERAGE HARD PRECISION = {:.4f}".format(avg_hard_score[0]))
    print("AVERAGE HARD F-MEASURE = {:.4f}".format(avg_hard_score[2]))

    print("\n*************************DISCOURSE***************************************\n")
    print("AVERAGE DISCOURSE RECAL =     {:.4f}".format(avg_discourse_score[1]))
    print("AVERAGE DISCOURSE PRECISION = {:.4f}".format(avg_discourse_score[0]))
    print("AVERAGE DISCOURSE F-MEASURE = {:.4f}".format(avg_discourse_score[2]))

    print("\n*************************OVERALL*****************************************\n")
    print("AVERAGE RECAL =     {:.4f}".format(avg_all_score[1]))
    print("AVERAGE PRECISION = {:.4f}".format(avg_all_score[0]))
    print("AVERAGE F-MEASURE = {:.4f}".format(avg_all_score[2]))
    print("\n*************************************************************************\n")


