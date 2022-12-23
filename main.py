import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
from functools import partial
from nltk.stem import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

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
    return int(str.split('_')[1][0])

def find_interests_index(liste):
    """
    find all the index of interests(^(?i)interests?_?[0-9]?) in a list of words 
    """
    result = []
    for i in range(len(liste)):
        if re.search(r'^(?i)*?interests?_[0-9]?', liste[i]):
            result.append(i)
            if liste[i][0] == '*':
                liste[i] = liste[i][1:]

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

    words_list = list(map( lambda x: split_by_space(eliminate_MGMNP(eliminate_brackets(elimimate_equals(x))).strip()), phrases))
    indexes_liste = list(map(find_interests_index, words_list))

    # next, for each interest, we need to get the context information
    context_information = []
    labels = []

    for i in range(len(indexes_liste)):
        indexes = indexes_liste[i]
        for index in indexes:
            interest = words_list[i][index]
            class_label = get_interest_class(interest)
            words_list[i][index] = re.sub(r'_[0-9]', '', interest)

            # append to labels
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
    #  i. window_size = 1
    word_nb_1 = MultinomialNB()
    word_nb_1.fit(word_train_1_x, word_train_1_y)
    score_word_nb_1 = word_nb_1.score(word_test_1_x, word_test_1_y)

    #  ii. window_size = 2
    word_nb_2 = MultinomialNB()
    word_nb_2.fit(word_train_2_x, word_train_2_y)
    score_word_nb_2 = word_nb_2.score(word_test_2_x, word_test_2_y)

    #  iii. window_size = 3
    word_nb_3 = MultinomialNB()
    word_nb_3.fit(word_train_3_x, word_train_3_y)
    score_word_nb_3 = word_nb_3.score(word_test_3_x, word_test_3_y)

    #  iv. whole context
    word_nb_whole = MultinomialNB()
    word_nb_whole.fit(word_train_whole_x, word_train_whole_y)
    score_word_nb_whole = word_nb_whole.score(word_test_whole_x, word_test_whole_y)

    print("Naive Bayes Multinomial Model:")
    print("word_nb_1: ", score_word_nb_1)
    print("word_nb_2: ", score_word_nb_2)
    print("word_nb_3: ", score_word_nb_3)
    print("word_nb_whole: ", score_word_nb_whole)

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Naive Bayes Multinomial Model for word context")
    plt.plot([1, 2, 3, "+inf"], [score_word_nb_1, score_word_nb_2, score_word_nb_3, score_word_nb_whole], label="word_nb")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    #b. pos context
    #  i. window_size = 1
    pos_nb_1 = MultinomialNB()
    pos_nb_1.fit(pos_train_1_x, pos_train_1_y)
    score_pos_nb_1 = pos_nb_1.score(pos_test_1_x, pos_test_1_y) 

    #  ii. window_size = 2
    pos_nb_2 = MultinomialNB()
    pos_nb_2.fit(pos_train_2_x, pos_train_2_y)
    score_pos_nb_2 = pos_nb_2.score(pos_test_2_x, pos_test_2_y)
    
    #  iii. window_size = 3
    pos_nb_3 = MultinomialNB()
    pos_nb_3.fit(pos_train_3_x, pos_train_3_y)
    score_pos_nb_3 = pos_nb_3.score(pos_test_3_x, pos_test_3_y)

    #  iv. whole context
    pos_nb_whole = MultinomialNB()
    pos_nb_whole.fit(pos_train_whole_x, pos_train_whole_y)
    score_pos_nb_whole = pos_nb_whole.score(pos_test_whole_x, pos_test_whole_y)

    print("pos_nb_1: ", score_pos_nb_1)
    print("pos_nb_2: ", score_pos_nb_2)
    print("pos_nb_3: ", score_pos_nb_3)
    print("pos_nb_whole: ", score_pos_nb_whole)
    print("")
    
    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Naive Bayes Multinomial Model for pos context")
    plt.plot([1, 2, 3, "+inf"], [score_pos_nb_1, score_pos_nb_2, score_pos_nb_3, score_pos_nb_whole], label="pos_nb")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # 2. Decision Tree
    # a. word context
    #  i. window_size = 1
    word_dt_1 = DecisionTreeClassifier()
    word_dt_1.fit(word_train_1_x, word_train_1_y)
    score_word_dt_1 = word_dt_1.score(word_test_1_x, word_test_1_y)

    #  ii. window_size = 2
    word_dt_2 = DecisionTreeClassifier()
    word_dt_2.fit(word_train_2_x, word_train_2_y)
    score_word_dt_2 = word_dt_2.score(word_test_2_x, word_test_2_y)
    
    #  iii. window_size = 3
    word_dt_3 = DecisionTreeClassifier()
    word_dt_3.fit(word_train_3_x, word_train_3_y)
    score_word_dt_3 = word_dt_3.score(word_test_3_x, word_test_3_y)

    #  iv. whole context
    word_dt_whole = DecisionTreeClassifier()
    word_dt_whole.fit(word_train_whole_x, word_train_whole_y)
    score_word_dt_whole = word_dt_whole.score(word_test_whole_x, word_test_whole_y)
    
    print("Decision Tree Model:")
    print("word_dt_1: ", score_word_dt_1)
    print("word_dt_2: ", score_word_dt_2)
    print("word_dt_3: ", score_word_dt_3)
    print("word_dt_whole: ", score_word_dt_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Decision Tree Model for word context")
    plt.plot([1, 2, 3, "+inf"], [score_word_dt_1, score_word_dt_2, score_word_dt_3, score_word_dt_whole], label="word_dt")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    #b. pos context
    #  i. window_size = 1
    pos_dt_1 = DecisionTreeClassifier()
    pos_dt_1.fit(pos_train_1_x, pos_train_1_y)
    score_pos_dt_1 = pos_dt_1.score(pos_test_1_x, pos_test_1_y)

    #  ii. window_size = 2
    pos_dt_2 = DecisionTreeClassifier()
    pos_dt_2.fit(pos_train_2_x, pos_train_2_y)
    score_pos_dt_2 = pos_dt_2.score(pos_test_2_x, pos_test_2_y)
    
    #  iii. window_size = 3
    pos_dt_3 = DecisionTreeClassifier()
    pos_dt_3.fit(pos_train_3_x, pos_train_3_y)
    score_pos_dt_3 = pos_dt_3.score(pos_test_3_x, pos_test_3_y)
    

    #  iv. whole context
    pos_dt_whole = DecisionTreeClassifier()
    pos_dt_whole.fit(pos_train_whole_x, pos_train_whole_y)
    score_pos_dt_whole = pos_dt_whole.score(pos_test_whole_x, pos_test_whole_y)

    print("pos_dt_1: ", score_pos_dt_1)
    print("pos_dt_2: ", score_pos_dt_2)
    print("pos_dt_3: ", score_pos_dt_3)
    print("pos_dt_whole: ", score_pos_dt_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Decision Tree Model for pos context")
    plt.plot([1, 2, 3, "+inf"], [score_pos_dt_1, score_pos_dt_2, score_pos_dt_3, score_pos_dt_whole], label="pos_dt")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # 3. Random Forest
    #a. word context
    #  i. window_size = 1
    word_rf_1 = RandomForestClassifier()
    word_rf_1.fit(word_train_1_x, word_train_1_y)
    score_word_rf_1 = word_rf_1.score(word_test_1_x, word_test_1_y)

    #  ii. window_size = 2
    word_rf_2 = RandomForestClassifier()
    word_rf_2.fit(word_train_2_x, word_train_2_y)
    score_word_rf_2 = word_rf_2.score(word_test_2_x, word_test_2_y)

    #  iii. window_size = 3
    word_rf_3 = RandomForestClassifier()
    word_rf_3.fit(word_train_3_x, word_train_3_y)
    score_word_rf_3 = word_rf_3.score(word_test_3_x, word_test_3_y)

    #  iv. whole context
    word_rf_whole = RandomForestClassifier()
    word_rf_whole.fit(word_train_whole_x, word_train_whole_y)
    score_word_rf_whole = word_rf_whole.score(word_test_whole_x, word_test_whole_y)

    print("Random Forest Model:")
    print("word_rf_1: ", score_word_rf_1)
    print("word_rf_2: ", score_word_rf_2)
    print("word_rf_3: ", score_word_rf_3)
    print("word_rf_whole: ", score_word_rf_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Random Forest Model for word context")
    plt.plot([1, 2, 3, "+inf"], [score_word_rf_1, score_word_rf_2, score_word_rf_3, score_word_rf_whole], label="word_rf")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    #b. pos context
    #  i. window_size = 1
    pos_rf_1 = RandomForestClassifier()
    pos_rf_1.fit(pos_train_1_x, pos_train_1_y)
    score_pos_rf_1 = pos_rf_1.score(pos_test_1_x, pos_test_1_y)
    

    #  ii. window_size = 2
    pos_rf_2 = RandomForestClassifier()
    pos_rf_2.fit(pos_train_2_x, pos_train_2_y)
    score_pos_rf_2 = pos_rf_2.score(pos_test_2_x, pos_test_2_y)

    #  iii. window_size = 3
    pos_rf_3 = RandomForestClassifier()
    pos_rf_3.fit(pos_train_3_x, pos_train_3_y)
    score_pos_rf_3 = pos_rf_3.score(pos_test_3_x, pos_test_3_y)

    #  iv. whole context
    pos_rf_whole = RandomForestClassifier()
    pos_rf_whole.fit(pos_train_whole_x, pos_train_whole_y)
    score_pos_rf_whole = pos_rf_whole.score(pos_test_whole_x, pos_test_whole_y)
    
    print("pos_rf_1: ", score_pos_rf_1)
    print("pos_rf_2: ", score_pos_rf_2)
    print("pos_rf_3: ", score_pos_rf_3)
    print("pos_rf_whole: ", score_pos_rf_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("Random Forest Model for pos context")
    plt.plot([1, 2, 3, "+inf"], [score_pos_rf_1, score_pos_rf_2, score_pos_rf_3, score_pos_rf_whole], label="pos_rf")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


    # 4. svm
    # a. word context
    #  i. window_size = 1
    word_svm_1 = svm.SVC()
    word_svm_1.fit(word_train_1_x, word_train_1_y)
    score_word_svm_1 = word_svm_1.score(word_test_1_x, word_test_1_y)

    #  ii. window_size = 2
    word_svm_2 = svm.SVC()
    word_svm_2.fit(word_train_2_x, word_train_2_y)
    score_word_svm_2 = word_svm_2.score(word_test_2_x, word_test_2_y)

    #  iii. window_size = 3
    word_svm_3 = svm.SVC()
    word_svm_3.fit(word_train_3_x, word_train_3_y)
    score_word_svm_3 = word_svm_3.score(word_test_3_x, word_test_3_y)

    #  iv. whole context
    word_svm_whole = svm.SVC()
    word_svm_whole.fit(word_train_whole_x, word_train_whole_y)
    score_word_svm_whole = word_svm_whole.score(word_test_whole_x, word_test_whole_y)

    print("SVM Model:")
    print("word_svm_1: ", score_word_svm_1)
    print("word_svm_2: ", score_word_svm_2)
    print("word_svm_3: ", score_word_svm_3)
    print("word_svm_whole: ", score_word_svm_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("SVM Model for word context")
    plt.plot([1, 2, 3, "+inf"], [score_word_svm_1, score_word_svm_2, score_word_svm_3, score_word_svm_whole], label="word_svm")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    #b. pos context
    #  i. window_size = 1
    pos_svm_1 = svm.SVC()
    pos_svm_1.fit(pos_train_1_x, pos_train_1_y)
    score_pos_svm_1 = pos_svm_1.score(pos_test_1_x, pos_test_1_y)
    
    #  ii. window_size = 2
    pos_svm_2 = svm.SVC()
    pos_svm_2.fit(pos_train_2_x, pos_train_2_y)
    score_pos_svm_2 = pos_svm_2.score(pos_test_2_x, pos_test_2_y)

    #  iii. window_size = 3
    pos_svm_3 = svm.SVC()
    pos_svm_3.fit(pos_train_3_x, pos_train_3_y)
    score_pos_svm_3 = pos_svm_3.score(pos_test_3_x, pos_test_3_y)

    #  iv. whole context
    pos_svm_whole = svm.SVC()
    pos_svm_whole.fit(pos_train_whole_x, pos_train_whole_y)
    score_pos_svm_whole = pos_svm_whole.score(pos_test_whole_x, pos_test_whole_y)

    print("pos_svm_1: ", score_pos_svm_1)
    print("pos_svm_2: ", score_pos_svm_2)
    print("pos_svm_3: ", score_pos_svm_3)
    print("pos_svm_whole: ", score_pos_svm_whole)
    print("")
    
    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("SVM Model for pos context")
    plt.plot([1, 2, 3, "+inf"], [score_pos_svm_1, score_pos_svm_2, score_pos_svm_3, score_pos_svm_whole], label="pos_svm")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # 5. multi-layer perceptron
    # a. word context
    # i. window_size = 1
    word_mlp_1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    word_mlp_1.fit(word_train_1_x, word_train_1_y)
    score_word_mlp_1 = word_mlp_1.score(word_test_1_x, word_test_1_y)

    #  ii. window_size = 2
    word_mlp_2 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    word_mlp_2.fit(word_train_2_x, word_train_2_y)
    score_word_mlp_2 = word_mlp_2.score(word_test_2_x, word_test_2_y)

    #  iii. window_size = 3
    word_mlp_3 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    word_mlp_3.fit(word_train_3_x, word_train_3_y)
    score_word_mlp_3 = word_mlp_3.score(word_test_3_x, word_test_3_y)

    #  iv. whole context
    word_mlp_whole = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    word_mlp_whole.fit(word_train_whole_x, word_train_whole_y)
    score_word_mlp_whole = word_mlp_whole.score(word_test_whole_x, word_test_whole_y)

    print("MLP Model:")
    print("word_mlp_1: ", score_word_mlp_1)
    print("word_mlp_2: ", score_word_mlp_2)
    print("word_mlp_3: ", score_word_mlp_3)
    print("word_mlp_whole: ", score_word_mlp_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("MLP Model for word context")
    plt.plot([1, 2, 3, "+inf"], [score_word_mlp_1, score_word_mlp_2, score_word_mlp_3, score_word_mlp_whole], label="word_mlp")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    #b. pos context
    #  i. window_size = 1
    pos_mlp_1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    pos_mlp_1.fit(pos_train_1_x, pos_train_1_y)
    score_pos_mlp_1 = pos_mlp_1.score(pos_test_1_x, pos_test_1_y)

    #  ii. window_size = 2
    pos_mlp_2 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    pos_mlp_2.fit(pos_train_2_x, pos_train_2_y)
    score_pos_mlp_2 = pos_mlp_2.score(pos_test_2_x, pos_test_2_y)
    
    #  iii. window_size = 3
    pos_mlp_3 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    pos_mlp_3.fit(pos_train_3_x, pos_train_3_y)
    score_pos_mlp_3 = pos_mlp_3.score(pos_test_3_x, pos_test_3_y)

    #  iv. whole context
    pos_mlp_whole = MLPClassifier(max_iter=1000, hidden_layer_sizes=(10,))
    pos_mlp_whole.fit(pos_train_whole_x, pos_train_whole_y)
    score_pos_mlp_whole = pos_mlp_whole.score(pos_test_whole_x, pos_test_whole_y)

    print("pos_mlp_1: ", score_pos_mlp_1)
    print("pos_mlp_2: ", score_pos_mlp_2)
    print("pos_mlp_3: ", score_pos_mlp_3)
    print("pos_mlp_whole: ", score_pos_mlp_whole)
    print("")

    # plot the results of different window size
    plt.figure(figsize=(10, 5))
    plt.title("MLP Model for pos context")
    plt.plot([1, 2, 3, "+inf"], [score_pos_mlp_1, score_pos_mlp_2, score_pos_mlp_3, score_pos_mlp_whole], label="pos_mlp")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    hidden_nodes_nums_word = []
    scores_word_mlp_1 = []
    for i in range(10,101,10):
        word_mlp_1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(i,))
        word_mlp_1.fit(word_train_1_x, word_train_1_y)
        score_word_mlp_1 = word_mlp_1.score(word_test_1_x, word_test_1_y)
        hidden_nodes_nums_word.append(i)
        scores_word_mlp_1.append(score_word_mlp_1)

    plt.figure(figsize=(10, 5))
    plt.title("MLP Model for word context")
    plt.plot(hidden_nodes_nums_word, scores_word_mlp_1, label="word_mlp")
    plt.xlabel("hidden nodes number")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    hidden_nodes_nums_pos = []
    scores_pos_mlp_1 = []
    for i in range(10,101,10):
        pos_mlp_1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(i,))
        pos_mlp_1.fit(pos_train_1_x, pos_train_1_y)
        score_pos_mlp_1 = pos_mlp_1.score(pos_test_1_x, pos_test_1_y)
        hidden_nodes_nums_pos.append(i)
        scores_pos_mlp_1.append(score_pos_mlp_1)

    plt.figure(figsize=(10, 5))
    plt.title("MLP Model for word context")
    plt.plot(hidden_nodes_nums_pos, scores_pos_mlp_1, label="pos_mlp")
    plt.xlabel("hidden nodes number")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

 

    
