#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

from qa_engine.base import QABase



# get the root node == head word node of the sentence
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None
    
def find_node(word, graph, lmtzr):
    for node in graph.nodes.values():
        sword = node['word']
        tag = node["tag"]
        if sword == None:
            continue
        if tag.startswith("V"):
            sword = lmtzr.lemmatize(sword,'v')
        else:
            sword = lmtzr.lemmatize(sword,'n')
        if sword == word:
            return node
    return None
    
def get_dependents(node, graph):
    #print("Graph nodes")
    #print(graph.nodes)
    results = []
    # print("Node dep:", node["word"])
    # print(node["deps"]) 
    for item in node["deps"]:
        # print("item:")
        # print(item)
        address = node["deps"][item][0] 
        #print("address:")
        #print(address)
        dep = graph.nodes[address]
        # print("dep")
        # print(dep)
        results.append(dep)
        results = results + get_dependents(dep, graph)
    #print("Results:")
    #print(results)
    #print(results)
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
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

def find_answer(qgraph, sgraph ,lmtzr, q_start):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    tag = qmain["tag"]
    if tag.startswith("V"):
        qword = lmtzr.lemmatize(qword,'v')
    else:
        qword = lmtzr.lemmatize(qword,'n')

    snode = find_node(qword, sgraph, lmtzr)

    print("Qmain word:")
    print(qword)

    # if snode is None means we didnt find the right recall senetence with key root in question
    if snode == None:
        return None
    else:
        print("Snode word:")
        print(snode["word"])
        
    answer = None

    if q_start == "where":
        answer = traverse_dep_nodes(sgraph, snode, "nmod")
    elif q_start == "what":
        answer = traverse_dep_nodes(sgraph, snode, "dobj")
    elif q_start == "who":
        answer = traverse_dep_nodes(sgraph, snode, "nsubj")
        # if the qmain word cant find the answer, try to search from its conjunctions 
        if snode['rel'] == 'conj':
            head_address = snode["head"]
            snode = sgraph.nodes[head_address]
            answer = traverse_dep_nodes(sgraph, snode, "nsubj")
            
    elif q_start == "when":
        answer = traverse_dep_nodes(sgraph, snode, "nmod")
    elif q_start == "why":
        pass
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
    #print("Story:")
    #print(story)
    # get the dependency graph of the first question
    qgraph = q["dep"]
    print("Dependency graph")
    print(qgraph)
    #print("qgraph:", qgraph)

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

