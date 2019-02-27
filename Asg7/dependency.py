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
        

    if q_start == "where":
        # head is the root of the subtree, we want subtree that nodes are directly connected to the main verb
        for node in sgraph.nodes.values():
            #print("node[head]=", node["head"])
            # meaning this subtree sentence has the same node as head
            if node.get('head', None) == snode["address"]:
                #print(node["word"], node["rel"])
                if node['rel'] == "nmod":
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    # print("DEPS")
                    # print(deps)
                    return " ".join(dep["word"] for dep in deps)
    elif q_start == "what":
        for node in sgraph.nodes.values():
            if node.get('head', None) == snode["address"]:
                # only gets the first sentence that meets the dependency
                # if qgraph.nodes[2]["word"] == "did":
                #     if node['rel'] == "nmod":
                #         deps = get_dependents(node, sgraph)
                #         deps = sorted(deps+[node], key=operator.itemgetter("address"))
                #         return " ".join(dep["word"] for dep in deps)
                # else:
                if node['rel'] in ('dobj'):
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    return " ".join(dep["word"] for dep in deps)
    elif q_start == "who":
        for node in sgraph.nodes.values():
            if node.get('head', None) == snode["address"]:
                if node['rel'] == "nsubj":
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    return " ".join(dep["word"] for dep in deps)
    elif q_start == "when":
        for node in sgraph.nodes.values():
            if node.get('head', None) == snode["address"]:
                if node['rel'] == "nmod":
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    return " ".join(dep["word"] for dep in deps)

    elif q_start == "why":
        pass
    elif q_start == "how":
        pass
    elif q_start == "did":
        pass


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

