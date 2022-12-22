import re
import numpy as np
import pandas as pd
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def find_nom_groups(str):
    return re.findall(r'\[.*?\]', str)

def find_interests(str):
    # find all the index of interests in the phrase
    return re.findall(r'(?i)\*?interests?_?[0-9]?', str)

def eliminate_brackets(str):
    return re.sub(r'\[|\]', '', str)

def elimimate_equals(str):
    # eliminate ========.... at the beginning of the phrase
    return re.sub(r'^=+ ', '', str)

def split_by_space(str):
    return re.split(r'\s+', str)

def extract_word(str):
    return re.split(r'/', str)[0]

def extract_pos(str):
    return re.split(r'/', str)[1]


def get_interest_class(str):
    """
    Interest_n/{string} ---> n
    *Interest" ---> -1  
    """
    if str[0] == '*':
        return -1
    else:
        return int(str.split('_')[1])

def find_interests_index(liste):
    """
    find all the index of interests((?i)\*?interests?_?[0-9]?) in a list of words 
    for example:["yuyang", "giegie", "*interesT", "interest_1/NN"] ---> [2, 3]
    """
    result = [i for i, word in enumerate(liste) if re.search(r'(?i)\*?interests?_?[0-9]?', word)]
    if len(result) == 0:
        print("Xiong gie gie says: no interest, no people")
        return None

    return result

def get_interests_from_liste(index_liste, liste):
    return [liste[i] for i in index_liste]

# liang ge interest de qingkuang, zhuyi quchu *
def get_context(indexes, liste, window_size):
    pass

if __name__ == "__main__":
    count = 0
    phrases = []
    with open("corpus.txt") as file:
        for i in file :
            if i!="$$\n":
                phrases.append(i.strip())
    phrase_1 =phrases[0]
    phrase_2 = "====================================== [ odds/NNS ] and/CC [ ends/NNS ] ====================================== despite/IN [ growing/VBG Interest_1/NN ] in/IN [ the/DT environment/NN ] ,/, [ u.s./NP consumers/NNS ] have/VBP n't/RB shown/VBN [ much/JJ *IntErest/NN ] in/IN [ refillable/JJ packages/NNS ] for/IN [ household/NN products/NNS ] ./. "
    print(phrase_1)
    GNs = find_interests(phrase_2)
    print(GNs)
    print("word/pos extraction", extract_word("interests_1/NN"))
    print("word/pos extraction", extract_pos("interests_1/NN"))
    print("find index test", find_interests_index(["yuyang", "giegie", "*interesT", "interest_1/NN"]))


