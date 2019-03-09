from nltk.corpus import wordnet as wn
from qa_howard import get_synsets
story_id_raw1 = "'blog-06.vgl'"
story_id_raw0 = "{'fables-06.vgl', 'blogs-01.vgl', 'fables-02.vgl'}"
story_id_raw = "'blogs-06.vgl', 'blogs-05.vgl', 'blogs-04.vgl', 'blogs-01.vgl', 'blogs-02.vgl', 'blogs-03.vgl', 'blogs-07.vgl'"
l_story_id_str = story_id_raw1.strip('{}').split(', ')
print(l_story_id_str)
for story_id_str in l_story_id_str:
	print(story_id_str.strip("\'").split('.')[0])
