import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words = []
'''all the sentences in the pattern are taken and broken down into words 
- stemming is doen and they are append to this list'''
classes = [] # these will have the tags stores
words_tags_list = [] # attaches the word and tag in the form of tuple ('Hi there!' , 'greeting')
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
'''a variable is created and it tried to open the file
and and read the file but as it is in json format 
we need to use json.load to read anything which is in json'''

#function for appending stem_Words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

''' the var intents refers to the line no 17 and hold its cursor
near the curly bracket we need to go inside the intents'''
for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            words_tags_list.append((pattern_word, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(words_tags_list[0]) 
print(classes)  

def create_corpus(stem_words,classes):
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words , open('words.pkl' , 'w'))
    pickle.dump(classes , open('classes.pkl' , 'w'))

    return stem_words, classes
stem_words, classes  = create_corpus(stem_words,classes)

print(stem_words)
print(classes)

#creating a bag of words
training_data = []
num_of_tags = len(classes)
labels = [0]*num_of_tags # a list of 9 zero [0,0,0,0,0,0,0,0,0] the tags

#create the training_data
for word_tags in words_tags_list :
    bag_of_words =  []
    pattern_words = word_tags[0] #('hi' , 'greeting') this has the 'hi' word in it stored

    for word in pattern_words:
        index = pattern_words.index(word)
        word = stemmer.stem(word.lower())
        pattern_words[index] = word

    for word in stem_words :
        if word in pattern_words :
            bag_of_words.append(1)
        else :
            bag_of_words.append(0)
    print(bag_of_words)

# create label encoding
    label = list(labels)
    tag = word_tags[1] #('hi' , 'greeting') #greeting is being stored
    tag_index = classes.index(tag)#index position of the tag
    label[tag_index] = 1

    training_data.append([bag_of_words,label])

print(training_data[0])


def preprocess_train_data(training_data):
    training_data = np.array(training_data , dtype=object)
    train_x = list(training_data[:,0])#all the rows and the first column
    train_y = list(training_data[:,1])#all the rows and the last lists

    print(train_x[0])
    print(train_y[0])

    return train_x , train_y

train_x , train_y = preprocess_train_data(training_data)
