import nltk
lemmatizer = nltk.wordnet.WordNetLemmatizer()
print(nltk.pos_tag(['The', 'narrator', 'saw', 'some', 'bright', 'flash']))
print(lemmatizer.lemm)