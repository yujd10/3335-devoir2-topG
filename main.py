import numpy as np
import pandas as pd
import nltk
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
from functools import partial
from sys import argv
from nltk.stem import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

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

def cut_context(index_liste,context,window_size):
    """
    cut the context into two parts, one is the left part, the other is the right part
    """
    new_indexes_list = [i for index in index_liste for i in index] 
    funct = partial(get_context,  window_size = window_size)
    return list(map(funct, zip(new_indexes_list, context)))

def tf_idf_vectorize_data(context):
    """
    split the data into training set and test set
    """
    new_context = []
    for i in range(len(context)):
        new_context.append(" ".join(context[i]))
    new_context = np.array(new_context)

    tfidf = TfidfVectorizer()
    res = tfidf.fit_transform(new_context)
    return res

def split_data(data, labels, test_size = 0.2):
    """
    split the data into training set and test set
    """
    x_and_y = (np.concatenate((data.toarray(),labels.reshape(-1,1)), axis = 1))
    train, test = sklearn.model_selection.train_test_split(x_and_y, test_size = test_size,random_state=42)
    return train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]


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
    
    #distinct two types of context_information: 1 word 2 pos 3 steamed word
    context_information_word = list(map(lambda x: list(map(extract_word, x)), context_information))
    context_information_pos = list(map(lambda x: list(map(extract_pos, x)), context_information))
    context_information_word = list(map(snowball_list_stemmer, context_information_word))

    # next, cut context_information according to window_size
    # case: window_size = 1
    context_word_1 = cut_context(indexes_liste, context_information_word, 1)
    context_pos_1 = cut_context(indexes_liste, context_information_pos, 1)
    tf_word_1 = tf_idf_vectorize_data(context_word_1)
    tf_pos_1 = tf_idf_vectorize_data(context_pos_1)

    word_train_1_x, word_train_1_y, word_test_1_x, word_test_1_y = split_data(tf_word_1, np.array(labels))
    pos_train_1_x, pos_train_1_y, pos_test_1_x, pos_test_1_y = split_data(tf_pos_1, np.array(labels))

    # case: window_size = 2
    context_word_2 = cut_context(indexes_liste, context_information_word, 2)
    context_pos_2 = cut_context(indexes_liste, context_information_pos, 2)
    tf_word_2 = tf_idf_vectorize_data(context_word_2)
    tf_pos_2 = tf_idf_vectorize_data(context_pos_2)

    word_train_2_x, word_train_2_y, word_test_2_x, word_test_2_y = split_data(tf_word_2, np.array(labels))
    pos_train_2_x, pos_train_2_y, pos_test_2_x, pos_test_2_y = split_data(tf_pos_2, np.array(labels))

    # case: window_size = 3
    context_word_3 = cut_context(indexes_liste, context_information_word, 3)
    context_pos_3 = cut_context(indexes_liste, context_information_pos, 3)
    tf_word_3 = tf_idf_vectorize_data(context_word_3)
    tf_pos_3 = tf_idf_vectorize_data(context_pos_3)

    word_train_3_x, word_train_3_y, word_test_3_x, word_test_3_y = split_data(tf_word_3, np.array(labels))
    pos_train_3_x, pos_train_3_y, pos_test_3_x, pos_test_3_y = split_data(tf_pos_3, np.array(labels))

    # case: whole context
    context_word_whole = context_information_word
    context_pos_whole = context_information_pos
    tf_word_whole = tf_idf_vectorize_data(context_word_whole)
    tf_pos_whole = tf_idf_vectorize_data(context_pos_whole)

    word_train_whole_x, word_train_whole_y, word_test_whole_x, word_test_whole_y = split_data(tf_word_whole, np.array(labels))
    pos_train_whole_x, pos_train_whole_y, pos_test_whole_x, pos_test_whole_y = split_data(tf_pos_whole, np.array(labels))

    #Models:
    # 1. MultinomialNB 
    #a. word context
    multinomialNB_word = MultinomialNB()
    multinomialNB_word.fit(word_train_x, word_train_y)
    socreNB_word=multinomialNB_word.score(word_test_x, word_test_y)

    #b. pos context
    multinomialNB_pos = MultinomialNB()
    multinomialNB_pos.fit(pos_train_x, pos_train_y)
    socreNB_pos=multinomialNB_pos.score(pos_test_x, pos_test_y)

    print("MultinomialNB word context accuracy: ", socreNB_word)
    print("MultinomialNB pos context accuracy: ", socreNB_pos)




            


