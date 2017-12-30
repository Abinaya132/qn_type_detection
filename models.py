import pandas as pd
import numpy as np
import nltk , re , itertools
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

#keras model 
import keras
from keras.models import model_from_json , Model
from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout , LSTM 
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


class get_data():
    def __init__(self):
        self.datapath = 'data/LabelledData.txt' 
    def load_data(self):
        self.data = pd.read_csv(self.datapath,sep=',,,',header=None)
        self.data = self.data.rename(columns={0:'Qn',1:'Type'})
        self.data.Type = self.data.Type.apply(lambda x:x.strip())
        self.data['Tokens'] = self.data.Qn.apply(lambda x:nltk.word_tokenize(x))
        return self.data 
    def get_x_y(self):
        self.data = self.load_data()
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.data.Type)
        self.inv_class_map = {i:x for i,x in enumerate(self.le.classes_)}
        return self.data.Qn.values , self.labels , self.inv_class_map 
    
#create a count vectorizer that takes a dist of the words and builds a Naive Bayes  model 

class count_vect_model():
    def __init__(self):
        self.x, self.y, self.inv_class_map = get_data().get_x_y()
        self.testsize = 0.33
        self.randomstate = 42
        self.count_vect = CountVectorizer()
        
    def get_countvect(self):
        self.X_train_counts = self.count_vect.fit_transform(self.x)
        self.clf_1 = MultinomialNB()
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(self.X_train_counts , self.y , 
                                                        test_size= self.testsize, random_state=self.randomstate)

        self.clf_1.fit(self.X_train,self.y_train)
        self.acc_ = cross_val_score(self.clf_1,self.X_test,self.y_test,cv=5) 
        return self.clf_1, self.acc_ , np.mean(self.acc_)
    

#creating the tf idf vectorizer based NB model. Does not do so well since the importance of the words like who why are 
#reduced as they occcur everytime , But in this context they should have high importance 

class tfidf_vect_model():
    def __init__(self):
        self.x, self.y, self.inv_class_map = get_data().get_x_y()
        self.testsize = 0.33
        self.randomstate = 42
        self.tfidf_vect = TfidfVectorizer()
        
    def get_tfidfvect(self):
        self.X_train_tfidf = self.tfidf_vect.fit_transform(self.x)
        self.clf_1 = MultinomialNB()
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(self.X_train_tfidf , self.y , 
                                                        test_size= self.testsize, random_state=self.randomstate)

        self.clf_1.fit(self.X_train,self.y_train)
        self.acc_ = cross_val_score(self.clf_1,self.X_test,self.y_test,cv=5) 
        return self.clf_1, self.acc_ , np.mean(self.acc_)
    
#create  a ML based model with one feature - if the headword - first word is a qn word and what is the qn word 

class get_model_feats():
    def __init__(self):
        self.qn_words = ['what','where','when','why','am','is','are','how','who','whom']
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.dict_vect = DictVectorizer()
        self.x, self.y, self.inv_class_map = get_data().get_x_y()
        self.testsize = 0.33
        self.randomstate = 42
        
    def create_features(self,qn):
        self.qn_token = nltk.word_tokenize(qn)
        self.qn_word_present = [qnword for qnword in self.qn_words if qnword in self.qn_token[1:]]
        self.qn_word_present = self.qn_word_present[0] if self.qn_word_present else 'None' 
        self.feature_dict = {}
        self.headword = self.qn_token[0].lower()
        self.feature_dict['Headword_qn'] = self.headword 
        return self.feature_dict


    def get_nb_model(self):
        self.x_features = map(self.create_features , self.x)
        self.X_train_feats = self.dict_vect.fit_transform(self.x_features)
        self.clf_4 = MultinomialNB()
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(self.X_train_feats , self.y , 
                                                test_size=self.testsize, random_state=self.randomstate)

        self.clf_4.fit(self.X_train , self.y_train)
        self.acc_ = cross_val_score(self.clf_4,self.X_test,self.y_test,cv=5) 
        return self.clf_4, self.acc_ , np.mean(self.acc_)
    

class kerasmodel():
    def __init__(self):
        self.x, self.y, self.inv_class_map = get_data().get_x_y()
        self.testsize = 0.33
        self.randomstate = 42
        self.embedding_dim = 256 
        self.num_filters = 512 
        self.drop = 0.6
        self.nb_epoch = 32
        self.batch_size = 8
        
    def clean_str(self , string):
        """
        Tokenization/string cleaning for datasets.
        Cleans the data and returns the cleaned string
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def process_data(self):
        self.x_text = [self.clean_str(sent) for sent in self.x]
        self.x_text = [s.split(" ") for s in self.x_text]
        
        #pads the words if len(tokens)<100 with the padding_word which is needed for symmetrical length 
        padding_word = '<PAD>'
        self.padded_sentences = []
        self.sequence_length =  100
        
        for i, sentence in enumerate(self.x_text):
            num_padding = self.sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            self.padded_sentences.append(new_sentence)
        
        #vocabulary for the initial embedding 
        self.word_counts = Counter(itertools.chain(*self.padded_sentences))
        self.vocabulary_inv = sorted(self.word_counts.keys())
        self.vocabulary = {x: i for i, x in enumerate(self.vocabulary_inv)}
        
        #convert the words to the initial embedding space and encoding the y's
        self.x = np.array([[self.vocabulary[word] for word in sentence] for sentence in self.padded_sentences])
        self.y = keras.utils.to_categorical(self.y)
        return self.vocabulary_inv , self.vocabulary , self.x , self.y  , self.inv_class_map 
    
    def get_sequence_length(self , x):
        return x.shape[1]
    
    def get_architecture(self,x,labels , vocabulary_inv,vocabulary,sequence_length):
        #the model takes the sentences and splits them into words . Applies a few conv layers on the words of the documents 
        #uses one dense layer and  a softmax layer to find the prob of occurence of each class
        #the model also trains the embedding of the words into a 256 dim vector 
        #sequence_length =  x.shape[1]
        vocabulary_size = len(vocabulary_inv)
        
        filter_sizes = [3,4,5]
        inputs = Input(shape=(self.sequence_length,), dtype='int32')
        embedding = Embedding(len(vocabulary) + 1,
                                    self.embedding_dim,
                                    input_length=self.sequence_length,
                                    trainable=True)(inputs)
        
        reshape = Reshape((sequence_length,self.embedding_dim,1))(embedding)
        
        conv_0 = Convolution2D(self.num_filters, filter_sizes[0], self.embedding_dim, border_mode='valid',
                               init='normal', activation='relu', dim_ordering='tf')(reshape)
        conv_1 = Convolution2D(self.num_filters, filter_sizes[1], self.embedding_dim, border_mode='valid',
                               init='normal', activation='relu', dim_ordering='tf')(reshape)
        conv_2 = Convolution2D(self.num_filters, filter_sizes[2], self.embedding_dim, border_mode='valid',
                               init='normal', activation='relu', dim_ordering='tf')(reshape)

        maxpool_0 = MaxPooling2D(pool_size=(self.sequence_length - filter_sizes[0] + 1, 1), 
                                 strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
        maxpool_1 = MaxPooling2D(pool_size=(self.sequence_length - filter_sizes[1] + 1, 1), 
                                 strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
        maxpool_2 = MaxPooling2D(pool_size=(self.sequence_length - filter_sizes[2] + 1, 1), strides=(1,1),
                                 border_mode='valid', dim_ordering='tf')(conv_2)

        merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)

        flatten = Flatten()(merged_tensor)

        dropout = Dropout(self.drop)(flatten)

        dense =  Dense(output_dim=128)(dropout)

        output = Dense(output_dim=self.y[0].shape[0], activation='softmax')(dense)

        self.model2 = Model(input=inputs, output=output)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        self.model2.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model2 
    
    def run_keras_model(self):
        
        self.vocabulary_inv , self.vocabulary , self.x , self.y , self.inv_class_map  = self.process_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                    test_size=self.testsize, random_state=self.randomstate)
        self.sequence_length = self.get_sequence_length(self.x)
        
        self.model2 = self.get_architecture(self.x, self.y, self.vocabulary_inv, self.vocabulary , self.sequence_length)
        
        self.model2.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, 
                   verbose=1, validation_data=(self.X_test, self.y_test))  # starts training
        # serialize model to JSON
        model_json = self.model2.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model2.save_weights("model.h5")
        print("Saved model to disk")
        
        return self.model2
    
def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


class test_prediction():
    def __init__(self,train=False):
        self.km =  kerasmodel()
        self.vocabulary_inv , self.vocabulary , self.x , self.y ,self.inv_class_map   = self.km.process_data()
        self.sequence_length = self.km.get_sequence_length(self.x)
        if train:
            self.model = kerasmodel().run_keras_model() 
        else:
            self.model = load_model() #if the model is pre loaded 
    
    def get_predictions(self,test_x_):
        self.x_text = [self.km.clean_str(sent) for sent in test_x_]
        self.x_text = [s.split(" ") for s in self.x_text]
        padding_word = '<PAD>'
        self.padded_sentences = []
        for i, sentence in enumerate(self.x_text):
            num_padding = self.sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            self.padded_sentences.append(new_sentence)

        self.test_x = np.array([[self.vocabulary[word] if word in self.vocabulary else 129 for word in sentence ] \
                                for sentence in self.padded_sentences])
        return zip(test_x_ , [self.inv_class_map[x.argmax()] for x in self.model.predict(self.test_x)])