# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

# print(wordnet.synsets('room'))

syn_arr = wordnet.synsets('paint can')
print(syn_arr)
print(syn_arr[0].definition())



from textblob import Word
w = Word("paper clip")
print(w.definitions)

# dog
# airplane
# automobile
# bird
# cat
# deer
# frog
# ship
# horse
# truck

# 10 classes