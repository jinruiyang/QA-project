def get_root(graph):
    for item in graph.nodes.values():
        if item['rel'] == 'root':
            return item

def get_nsubj(node, graph):
    dep = None
    for item in node['deps']:
        if item == 'nsubj':
            address = node['deps'][item][0]
            dep = graph.nodes[address]['word']
    return dep

def get_words_from_graph(graph):
    words = []
    for item in graph.nodes.values():
        word = item['word']
        if word:
            words.append(word)
    return words

def get_words_tag_from_graph(graph):
    words = []
    for item in graph.nodes.values():
        word = item['word']
        if word:
            words.append((word, item['tag']))
    return words



def get_list_of_address(node):
    list_address = []
    list_address.append(node['address'])
    for item in node['deps']:
        address = node['deps'][item]
        for add in address:
            list_address.append(add)

    return list_address

def get_list_of_address_into_string(node, graph):
    words = []
    list_address = get_list_of_address(node)
    for address in sorted(list_address):
        word = graph.nodes[address]['word']
        if word:
            words.append(word)

    #       print(words)
    #none type found?
    return " ".join(word for word in words)


#this is getting a keyword from a question and this is reliable.
def get_keyword_from_question(graph):
    target_word = get_root(graph)
    word = target_word['lemma']

    if target_word['tag'] == 'WP':
        #print(graph)
        #for item in root['deps']:
        #    if item == 'nsubj':
        if 'root' in target_word['deps']:
            address = target_word['deps']['root'][0]
            target_word = graph.nodes[address]
        elif 'dep' in target_word['deps']:
            address = target_word['deps']['dep'][0]
            target_word = graph.nodes[address]
        else:
            address = target_word['deps']['nsubj'][0]
            target_word = graph.nodes[address]


        if target_word['word'] == 'did':
            for w in graph.nodes[address]['deps']:
                if w != 'punct':
                    add = graph.nodes[address]['deps'][w][0]
                    word = graph.nodes[add]['lemma']
                    break

    if 'nmod' in target_word['deps']:
        address = target_word['deps']['nmod'][0]
        target_word = graph.nodes[address]
        if 'acl' in target_word['deps']:
            address = target_word['deps']['acl'][0]
            word = graph.nodes[address]['lemma']
        elif 'nmod:poss' in target_word['deps']:
            address = target_word['deps']['nmod:poss'][0]
            word = graph.nodes[address]['lemma']
        elif 'amod' in target_word['deps']:
            address = target_word['deps']['amod'][0]
            word = graph.nodes[address]['lemma']
        else:
            word = graph.nodes[address]['lemma']
    #else:
    #    word = target_word['lemma']

    else:
        for item in target_word['deps']:
            if item == 'xcomp':
                address = target_word['deps'][item][0]
                word = graph.nodes[address]['lemma']
                break


    return word

def get_keyword_from_sentence(graph, first_w, main_word):
    keyword = None
    if not main_word:
        return None

    if first_w == 'who' or first_w == 'Who':
        for item in graph.nodes.values():
            if item['rel'] == 'root':
                keyword = get_nsubj(item, graph)

    elif first_w == 'what' or first_w == 'What':
        address = None
        for item in graph.nodes.values():
            if item['word']:
                #get address of the main_word
                if main_word == item['word'] or main_word in item['word'] or main_word == item['lemma']:
                    address = item['address']

        #look for the one that has nsubj as address of the main_address
        if address:
            target_word = None
            for item in graph.nodes.values():
                if item['word'] == main_word or item['lemma'] == main_word:
                    target_word = item
                    break

            if target_word:
                if 'nmod' not in target_word['deps']:
                    if 'dobj' in target_word['deps']:
                        address = target_word['deps']['dobj'][0]
                        target_word = graph.nodes[address]
                        keyword = get_list_of_address_into_string(target_word, graph)

                if target_word['rel'] == 'xcomp':
                    for item in graph.nodes.values():
                        if item['address'] != target_word['address'] and item['rel'] == 'xcomp':
                            keyword = get_list_of_address_into_string(item, graph)
                            break
            #print(keyword)

    elif first_w == 'why' or first_w == 'Why':
        words = get_words_from_graph(graph)
        if 'because' in words:
            index = words.index('because')
            keyword = ' '.join(word for word in words[index:])

    #do the why part

    #do the when part

    #do the where part

    return keyword


def turn_in_sentence(graph):
    return " ".join(item["word"] for item in graph.nodes.values() if item["word"])
