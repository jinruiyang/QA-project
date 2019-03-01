import nltk
import qa_howard


question = 'What did the Serpent spat?'
normalized_words = qa_howard.normalize_to_words(question)
q_start_word = normalized_words[0]
normalized_words = normalized_words[1:]  # remove start_word
key_words, key_vb = qa_howard.lemmatize_vb(normalized_words)

print(q_start_word)
print(normalized_words)
print(key_words)
print(key_vb)

sentence = 'The serpent spat some poison in the drinking vessel.'

normalized_sent_words = qa_howard.normalize_to_words(sentence)
sent_word_tag = nltk.pos_tag(qa_howard.tokenize_words(sentence))
print(sent_word_tag)
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove('from')
normalized_sent_word_tag = [(word.lower(), tag) for word, tag in sent_word_tag if word not in stopwords]
key_sent_words = qa_howard.lemmatize_vb_with_tag(normalized_sent_word_tag)
print(key_sent_words)
# lemmatizer = nltk.wordnet.WordNetLemmatizer()
# words = nltk.word_tokenize('bean bag bullets were fired at the rioters')
# print(nltk.pos_tag(words))
# print(lemmatizer.lemmatize('fired'))