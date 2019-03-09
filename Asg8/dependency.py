#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

from qa_engine.base import QABase
verbose = True

# get the root node == head word node of the sentence
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None


def get_words_from_graph(graph):
    words = []
    for item in graph.nodes.values():
        word = item['word']
        if word:
            words.append(word)
    return words


def find_node(word, graph, lmtzr):
    for node in graph.nodes.values():
        sword = node['word']
        tag = node["tag"]
        if sword == None:
            continue
        if tag.startswith("V"):
            sword = lmtzr.lemmatize(sword, 'v')
            # because the error of wordnet
            if sword == "felt":
                sword = "feel"
            elif sword == "gnawing":
                sword = "gnaw"
            
        else:
            sword = lmtzr.lemmatize(sword, 'n')
        if sword == word:
            return node
    return None


def get_dependents(node, graph):
    # print("Graph nodes")
    # print(graph.nodes)
    results = []
    # print("Node dep:", node["word"])
    # print(node["deps"])
    for item in node["deps"]:
        # print("item:")
        # print(item)
        address = node["deps"][item][0]
        # print("address:")
        # print(address)
        dep = graph.nodes[address]
        # print("dep")
        # print(dep)
        results.append(dep)
        results = results + get_dependents(dep, graph)
    # print("Results:")
    # print(results)
    # print(results)
    return results




"""
This method is used to traverse all nodes that is the child of the node
that has certain relation to the snode.
"""


def traverse_dep_nodes(sgraph, snode, rel_to_head):
    for node in sgraph.nodes.values():
        # meaning this node is the child of its head node
        if node.get('head', None) == snode["address"]:
            # define the relation with the snode we want to find for each question
            if node['rel'] == rel_to_head:
                deps = get_dependents(node, sgraph)
                deps = sorted(deps + [node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

def lemmatize_qmain(qmain, qword, lmtzr):
    tag = qmain["tag"]

    if tag.startswith("V"):
        qword = lmtzr.lemmatize(qword, 'v')
    else:
        qword = lmtzr.lemmatize(qword, 'n')

        if qword == "standing":
            return "stand"

    return qword


def find_answer(qgraph, sgraph, lmtzr, q_start):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    # the relation between q_start and the qmain word
    q_start_rel = qgraph.nodes[1]["rel"]
    answer = None
    
    qword = lemmatize_qmain(qmain, qword, lmtzr)
    snode = find_node(qword, sgraph, lmtzr)

    if verbose:
        with open("tmp.txt", "a") as f:
            f.write("Qmain word:\n")
            f.write(qword+"\n")

    # if snode is None means we didnt find the right recall senetence with key root in question
    if snode == None:
        return None
    else:
        if verbose:
            with open("tmp.txt", "a") as f:
                f.write("Snode word:\n")
                f.write(snode["word"]+"\n")



    if q_start == "where":
        answer = traverse_dep_nodes(sgraph, snode, "nmod")

        if not answer and snode['rel'] == "conj":
            head_address = snode["head"]
            snode = sgraph.nodes[head_address]
            answer = traverse_dep_nodes(sgraph, snode, "nmod")


    elif q_start == "what":
        # print(q_start_rel)
        if q_start_rel == "dobj":
            answer = traverse_dep_nodes(sgraph, snode, "dobj")

            if not answer and snode['rel'] == 'conj':
                head_address = snode["head"]
                snode = sgraph.nodes[head_address]
                answer = traverse_dep_nodes(sgraph, snode, "dobj")

        elif q_start_rel in ("nsubj", "nsubjpass"):
            if q_start_rel == "nsubj":
                answer = traverse_dep_nodes(sgraph, snode, "nsubj")
            elif q_start_rel == "nsubjpass":
                answer = traverse_dep_nodes(sgraph, snode, "nsubjpass")
                if not answer: 
                    answer = traverse_dep_nodes(sgraph, snode, "nsubj")

            if not answer and snode['rel'] == 'conj':
                head_address = snode["head"]
                snode = sgraph.nodes[head_address]
                answer = traverse_dep_nodes(sgraph, snode, "nsubj")

        else:
            answer = traverse_dep_nodes(sgraph, snode, "dobj")

        if not answer:
            answer = traverse_dep_nodes(sgraph, snode, "xcomp")
        if not answer:
            answer = traverse_dep_nodes(sgraph, snode, "ccomp")

    elif q_start == "who":
        if q_start_rel in ("nsubj", "nsubjpass"):
            if q_start_rel == "nsubj":
                answer = traverse_dep_nodes(sgraph, snode, "nsubj")
            elif q_start_rel == "nsubjpass":
                answer = traverse_dep_nodes(sgraph, snode, "nsubjpass")
                if not answer: answer = traverse_dep_nodes(sgraph, snode, "nsubj")
        else:        
            answer = traverse_dep_nodes(sgraph, snode, q_start_rel)

        if not answer and q_start_rel=="dep":
            answer = traverse_dep_nodes(sgraph, snode, "nsubj")

        if not answer and q_start_rel=="dep":
            answer = traverse_dep_nodes(sgraph, snode, "dobj")

        if not answer:
            # if the qmain word cant find the answer, try to search from its conjunctions
            if snode['rel'] == 'conj':
                head_address = snode["head"]
                snode = sgraph.nodes[head_address]
                answer = traverse_dep_nodes(sgraph, snode, "nsubj")

    elif q_start == "when":
        answer = traverse_dep_nodes(sgraph, snode, "nmod")
    elif q_start == "why":
        words = get_words_from_graph(sgraph)
        #print("---------------------------")
        #print(words)
        keys1 = [['in', 'order', 'for'], ['in', 'order', 'to']]
        for key in keys1:
            if key in words:
                index = words.index(key[0])
                return ' '.join(word for word in words[index:])
                #print(' '.join(word for word in words[index:]))
                #print("11111111111")
        keys2 = ['because', 'so', 'for', 'to']
        for key in keys2:
            if key in words:
                index = words.index(key)
                return ' '.join(word for word in words[index:])
                #print(' '.join(word for word in words[index:]))
    elif q_start == "how":
        pass
    elif q_start == "did":
        pass

    return answer


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("fables-01-1")
    print("Question:")
    print(q)
    story = driver.get_story(q["sid"])
    # print("Story:")
    # print(story)
    # get the dependency graph of the first question
    qgraph = q["dep"]
    print("Dependency graph")
    print(qgraph)
    # print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    sgraph = story["sch_dep"][1]
    print("sgraph:")
    print(story["sch_dep"])

    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        print(node)
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()

    answer = find_answer(qgraph, sgraph)
    print("answer:", answer)