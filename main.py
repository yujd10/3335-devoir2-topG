import re
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def find_nom_groups(str):
    return re.findall(r'\[.*?\]', str)

def find_interests(str):
    return re.findall(r'(?i)\*?interests?_?[0-9]?', str)

if __name__ == "__main__":
    count = 0
    phrases = []
    with open("corpus.txt") as file:
        for i in file :
            if i!="$$\n":
                phrases.append(i)
    phrase_1 =phrases[0]
    phrase_2 = "====================================== [ odds/NNS ] and/CC [ ends/NNS ] ====================================== despite/IN [ growing/VBG Interest_1/NN ] in/IN [ the/DT environment/NN ] ,/, [ u.s./NP consumers/NNS ] have/VBP n't/RB shown/VBN [ much/JJ *IntErest/NN ] in/IN [ refillable/JJ packages/NNS ] for/IN [ household/NN products/NNS ] ./. "
    print(phrase_1)
    GNs = find_interests(phrase_2)
    print(GNs)