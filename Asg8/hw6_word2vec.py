import zipfile, os
import re, nltk
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from word2vec_extractor import Word2vecExtractor
from dependency_demo_stub import read_dep_parses, find_main


###############################################################################
## Utility Functions ##########################################################
###############################################################################

# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':}, 'fables-01-2': {...}, ...}
def getQA(filename):
    content = open(filename, 'rU', encoding='latin1').read()
    question_dict = {}
    for m in re.finditer(r"QuestionID:\s*(?P<id>.*)\nQuestion:\s*(?P<ques>.*)\n(Answer:\s*(?P<answ>.*)\n){0,1}Difficulty:\s*(?P<diff>.*)\nType:\s*(?P<type>.*)\n+", content):
        qid = m.group("id")
        question_dict[qid] = {}
        question_dict[qid]['Question'] = m.group("ques")
        question_dict[qid]['Answer'] = m.group("answ")
        question_dict[qid]['Difficulty'] = m.group("diff")
        question_dict[qid]['Type'] = m.group("type")
    return question_dict

def get_data_dict(fname):
    data_dict = {}
    data_types = ["story", "sch"]
    parser_types = ["par", "dep"]
    for dt in data_types:
        data_dict[dt] = read_file(fname + "." + dt)
        for tp in parser_types:
            data_dict['{}.{}'.format(dt, tp)] = read_file(fname + "." + dt + "." + tp)
    return data_dict

# Read the file from disk
# filename can be fables-01.story, fables-01.sch, fables-01-.story.dep, fables-01.story.par
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()   
    return text

###############################################################################
## Question Answering Functions Baseline ######################################
###############################################################################

def get_bow(text, stopwords):
    return set([token.lower() for token in nltk.word_tokenize(text) if token.lower() not in stopwords])
	
def find_phrase(qbow, sent):
    tokens = nltk.word_tokenize(sent)
    # Travel from the end to begin.
    for i in range(len(tokens) - 1, 0, -1):
        word = tokens[i]
        # If find a word that match the question,
        # return the phrase that behind that word.
        # For example, "lion" occur in the question,
        # So we will return "want to eat the bull" which originally might look like this "... the lion want to eat the bull" 
        if word in qbow:
            return " ".join(tokens[i+1:])
	
# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def get_candidate_answers(question, text, W2vecextractor, q_verb, sgraphs, useWord2Vec = False, useVerb = True):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    
    # Collect all the candidate answers
    candidate_answers = []

    if(useWord2Vec==True and useVerb == False):
        q_feat = W2vecextractor.sent2vec(question)
        sentences = nltk.sent_tokenize(text)
        for i in range(0, len(sentences)):
            sent = sentences[i]
            a_feat = W2vecextractor.sent2vec(sent)
            dist = cosine_similarity(q_feat, a_feat) #calculate cosine similarity between the question and the candidate answer
            candidate_answers.append((dist, i, sent))
            #print("distance: "+str(dist)+"\t sent: "+sent)

    if(useWord2Vec==True and useVerb == True):
        #print(q_verb)
        q_feat = W2vecextractor.word2v(q_verb)
        sentences = nltk.sent_tokenize(text)
        for i in range(0, len(sentences)):
            sent = sentences[i]
            s_verb = find_main(sgraphs[i])['word']
            #print(s_verb)
            a_feat = W2vecextractor.word2v(s_verb)
            dist = cosine_similarity(q_feat, a_feat)   #calculate cosine similarity between the main verbs in the question and the candidate answer
            candidate_answers.append((dist, i, sent))
            #print("distance: "+str(dist)+"\t sent: "+sent)

    else:
        qbow = get_bow(question, stopwords)
        sentences = nltk.sent_tokenize(text)
        for i in range(0, len(sentences)):
            sent = sentences[i]
            # A list of all the word tokens in the sentence
            sbow = get_bow(sent, stopwords)
        
            # Count the # of overlapping words between the Q and the A
            # & is the set intersection operator
            overlap = len(qbow & sbow)
        
            candidate_answers.append((overlap, i, sent))
        
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    # Make sure to check about whether the results are null.
    #if len(candidate_answers) > 0:
        #best_answer = sorted(candidate_answers, key=lambda x: x[0], reverse=True)[0][1]
        #best_answer = max(candidate_answers, key=lambda x: x[0])[1]
        #return best_answer 
    return sorted(candidate_answers, key=lambda x: x[0], reverse=True)
   


###############################################################################
## Chunking ###################################################################
###############################################################################

###############################################################################
## Constituency ###############################################################
###############################################################################



    
#######################################################################

if __name__ == '__main__':
    W2vecextractor = Word2vecExtractor()
    useWord2Vec = True  #Set to True to use word2vec instead of baseline 
    useVerb = False #Set to True to use word2vec for ROOT verb
    # Loop over the files in fables and blogs in order.


    if(useWord2Vec):
         if(useVerb):
             output_file = open("train_my_answers_w2v_verb.txt", "w", encoding="utf-8")
         else:
             output_file = open("train_my_answers_w2v.txt", "w", encoding="utf-8")

    else:
             output_file = open("train_my_answers.txt", "w", encoding="utf-8")

    cname_size_dict = OrderedDict();
    cname_size_dict.update({"fables":2})
    cname_size_dict.update({"blogs":1})
    for cname, size in cname_size_dict.items():
        for i in range (0, size):
            # File format as fables-01, fables-11
            fname = "{0}-{1:02d}".format(cname, i+1)
            #print("File Name: " + fname)
            data_dict = get_data_dict(fname)

            questions = getQA("{}.questions".format(fname))

            qgraphs = read_dep_parses(fname+".questions.dep")
            

            for j in range(0, len(questions)):
                qname = "{0}-{1}".format(fname, j+1)
                if qname in questions:
                    print("QuestionID: " + qname)
                    question = questions[qname]['Question']
                    print(question)
                    qtypes = questions[qname]['Type']
   
                    # Get the question dep graph
                    qgraph = qgraphs[i]

                    # Get main verb in the question
                    q_verb = find_main(qgraph)['word']
                    
                    answer = None
                    # qtypes can be "Story", "Sch", "Sch | Story"
                    for qt in qtypes.split("|"):
                        qt = qt.strip().lower()
                        # These are the text data where you can look for answers.
                        raw_text = data_dict[qt]
                        par_text = data_dict[qt + ".par"]
                        dep_text = data_dict[qt + ".dep"]
 
                        # get the applicable dep file for finding the answer
                        ans_dep_file = fname+"."+str(qt)+".dep"

                        # get the dep graphs for all sentences in the answer file
                        sgraphs = read_dep_parses(ans_dep_file)
                        
                        candidate_answers = get_candidate_answers(question, raw_text, W2vecextractor, q_verb, sgraphs, useWord2Vec, useVerb)
                        answer = candidate_answers[0][2]
                                            
                    print("Answer: " + str(answer))
                    print("")

                    # Save your results in output file.
                    output_file.write("QuestionID: {}\n".format(qname))
                    output_file.write("Answer: {}\n\n".format(answer))
    output_file.close()

                    



