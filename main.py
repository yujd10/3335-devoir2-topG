import numpy as np
import pandas as pd
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
from functools import partial
from sys import argv
from nltk.stem import SnowballStemmer

stoplist_path = "stoplist-english.txt"

def stoplists():
    stoplist = []
    with open(stoplist_path) as file:
        for i in file :
            stoplist.append(i.strip())
    return stoplist

def eliminate_brackets(str):
    return re.sub(r'\[|\]', '', str)

def elimimate_equals(str):
    # eliminate all occurences of ========.... in the str 
    return re.sub(r'=+', '', str)

def eliminate_MGMNP(str):
    return re.sub(r'MGMNP', '', str)

def split_by_space(str):
    return re.split(r'\s+', str)

def extract_word(str):
    return re.split(r'/', str)[0]

def extract_pos(str):
    try:
        return re.split(r'/', str)[1]
    except:
        print("error", str)
        return None

def get_interest_class(str):
    """
    Interest_n/{string} ---> n
    *Interest" ---> -1   (we do not need this anymore, so just ignore this line)
    """
    # if str[0] == '*':
    #     return -1
    # else:
    #     return int(str.split('_')[1][0])
    return int(str.split('_')[1][0])

def find_interests_index(liste):
    """
    find all the index of interests(^(?i)interests?_?[0-9]?) in a list of words 
    for example:["yuyang", "giegie", "*interesT/NN", "interest_1/NN"] ---> [2, 3]
    """
    # write a version using loop
    result = []
    for i in range(len(liste)):
        if re.search(r'^(?i)*?interests?_[0-9]?', liste[i]):
            result.append(i)
            if liste[i][0] == '*':
                liste[i] = liste[i][1:]
        # modify the second case in another funciton.


    if len(result) == 0:
        print("Xiong gie gie says: no interest, no people")
        return None

    return result


def get_context(index_liste, window_size):
    """
        get window_size words before and after the word at index in the liste
        return a list of these words
    """
    index = index_liste[0]
    liste = index_liste[1]
    result = []
    for i in range(index - window_size, index + window_size + 1):
        if i < 0 or i >= len(liste) or i == index:
            continue
        else:
            result.append(liste[i])
    return result

def eliminate_stopwords(liste):
    """
    eliminate a word if it contains a stopword 
    example :"qsdqfqefd now/RB dasdqwdfqwfd"->"qsdqfqefd dasdqwdfqwfd"
    example :"and/CC elsewhere/RB ,/, now/RB will/MD be/VB watching/VBG [ the/DT american/JJ presidency/NN ]"->" elsewhere/RB ,/,  watching/VBG [  american/JJ presidency/NN ]"
    """
    stoplist = stoplists()
    words = split_by_space(liste)
    result = []
    for word in words:
        if extract_word(word) not in stoplist:
            result.append(word)
    return " ".join(result)
    
def snowball_list_stemmer(liste):
    """
    stem a list of words
    """
    stemmer = SnowballStemmer("english")
    return list(map(stemmer.stem, liste))


if __name__ == "__main__":
    count = 0
    phrases = []
    with open("corpus.txt") as file:
        for i in file :
            if i!="$$\n":
                #stoplist elimination
                phrases.append(eliminate_stopwords(i.strip()))

    # print(phrases[0])
    # 大的思路是， eliminate_equals---->
    #             eliminate_brackets----->
    #              eliminate_MGMNP------>
    #              lower()????????????????????????????---->
    #            split_by_space-->
    #           find_interests_index

    words_list = list(map( lambda x: split_by_space(eliminate_MGMNP(eliminate_brackets(elimimate_equals(x))).strip()), phrases))
    indexes_liste = list(map(find_interests_index, words_list))

    print(indexes_liste[0])

    # next, for each interest, we need to get the context information
    context_information = []
    labels = []

    for i in range(len(indexes_liste)):
        indexes = indexes_liste[i]
        for index in indexes:
            interest = words_list[i][index]
            class_label = get_interest_class(interest)
            # eliminate the ...._{class_label}... part in the interest
            # for example: interests_1/NN ---> interests/NN
            words_list[i][index] = re.sub(r'_[0-9]', '', interest)

            # append to  labels
            labels.append(class_label)
        
        # append to context_information
        for j in range(len(indexes)):
            context_information.append(words_list[i])
    
#     #distinct two types of context_information: 1 word 2 pos
    context_information_word = list(map(lambda x: list(map(extract_word, x)), context_information))
    context_information_pos = list(map(lambda x: list(map(extract_pos, x)), context_information))

    # print("context_information_word", context_information_word[0:10])
    # print("context_information_pos", context_information_pos[0:10])

    steamed_information_word = list(map(snowball_list_stemmer, context_information_word))
    print("steamed_context_information_word",len(steamed_information_word))

    # next, cut context_information according to window_size
    # window_size = int(argv[1])
    window_size = 4
    new_indexes_list = [i for index in indexes_liste for i in index] 
    funct = partial(get_context,  window_size = window_size)
    steamed_context_word = list(map(funct, zip(new_indexes_list, steamed_information_word)))
    context_word = list(map(funct, zip(new_indexes_list, context_information_word)))
    context_pos = list(map(funct, zip(new_indexes_list, context_information_pos)))


    stemmed_context_word_ = []
    for i in range(len(steamed_context_word)):
        stemmed_context_word_.append(" ".join(steamed_context_word[i]))

    context_word_ = []
    for i in range(len(context_word)):
        context_word_.append(" ".join(context_word[i]))

    context_pos_ = []
    for i in range(len(context_pos)):
        context_pos_.append(" ".join(context_pos[i]))
    
    tfidf = TfidfVectorizer()
    tf_stemmed_context =tfidf.fit_transform(stemmed_context_word_)
    tf_context = tfidf.fit_transform(context_word_)
    tf_pos = tfidf.fit_transform(context_pos_)

    labels = np.array(labels)
    
    print(labels)

    #task: 1. combine train_x and labels
    #      2. split train_x and labels into train and test
    #      3. train the model
    #      4. test the model

    

    # print("steamed_context_word_tfidf", steamed_context_word_tfidf.shape)
    # print("context_word", context_word[0:10])
    # print("context_pos", context_pos[0:10])
    # vecotr_stemed_context_word = 



            


