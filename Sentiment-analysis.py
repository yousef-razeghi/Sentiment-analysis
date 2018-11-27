
#YOUSEF RAZEGHI-OZYEGIN UNIVERSITY-COMPUTER SCIENCE DEPARTMENT
from sklearn import datasets, model_selection
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import text, FeatureHasher
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier
# $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
def eval(cls,k,data,target, method=None, test_ratio=0.25):
    score = 0
    if method == 'kfold':
        kfold = model_selection.KFold(n_splits=k)
        for ind_train,ind_test in kfold.split(data):
            cls.fit(data[ind_train], target[ind_train])
            ypred = cls.predict(data[ind_test])
            score += cls.score(data[ind_test],target[ind_test])
    else:
        for I in range(k):
            xtrain,xtest,ytrain,ytest=model_selection.train_test_split(data,target,test_size=test_ratio)
            cls.fit(xtrain,ytrain)
            score += cls.score(xtest,ytest)
            end = time.time()
    return score/k
# $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
stop_words_list=[".","/","#","&lt","!","?","<","[","]","{","}",";",":","'","|",">","$","%","^","&","*",
                 "(",")","_","+","=","~","at first","at last","at that moment","after","as soon as",
                 "before","between","by","1","2","3","4","5","6","7","8","9","0",".","a","b","c",
                 "d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
                 "during","earlier","eventually","except","finally","following","for","from then on",
                 "in the meantime","in the end","in addition","not a moment too soon","now","next",
                 "next week","suddenly","shortly after that","then","therefor","tomorrow","yesterday",
                 "apple","@","and", "is", "that", "this", "are", "am", "was", "were", "me", "mine",
                 "them", "there", "here", "their", "then", "when", "on", "in", "with",
                 "without","time,person,year,way,day,thing","man","world","life","hand","part",
                 "child","eye","woman","place","work","week","case","point","government",
                 "company","number","group","problem","fact","be","have","do","say","get",
                 "make","go","know","take","see","come","think","look","want","give","use",
                 "find","tell","ask","work","seem","feel","try","leave","call","good","new",
                 "first","last","long","great","little","own","other","old","right","big",
                 "high","different","small","large","next","early","young","important","few",
                 "public","bad","same","able","to","of","in","for","on","with","at","by",
                 "from","up","about","into","over","after","beneath","under","above","the",
                 "and","a","that","it","not","he","as","you","this","but","his","they","her",
                 "she","or","an","will","my","one","all","would","there","their","Afghanistan",
                 "Albani","Algeria","Andorra","Angola","Antigua","Barbuda","Argentina","Armenia",
                 "Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh",
                 "Barbados","Belarus","Belgium","Belize","Benin","Bhutan","Bolivia","Bosnia and Herzegovina",
                 "Botswana","Brazil","Brunei","Bulgaria","Burkina Faso","Burma","Burundi","Cambodia",
                 "Cameroon","Canada","Cabo Verde","Central African Republic","Chad","Chile","China",
                 "Colombia","Comoros","Congo"," Democratic Republic of the Congo"," Costa RicaCote d",
                 "Ivoire","Croatia","Cuba","Curacao","Cyprus","Czechia","Denmark","Djibouti",
                 "Dominica","Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial",
                 "Guinea","Eritrea","Estonia","Ethiopia","Fiji","Finland","France","Gabon",
                 "Gambia","GeorgiaGermanyGhanaGreeceGrenada","Guatemala","Guinea","Guinea-Bissau",
                 "Guyana","Haiti","Holy See","Honduras","Hong Kong","Hungary","Iceland","India",
                 "Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan",
                 "Kazakhstan","Kenya","Kiribati","Korea"," NorthKorea"," SouthKosovo","Kuwait",
                 "Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein",
                 "Lithuania","Luxembourg","Macau","Macedonia","Madagascar","Malawi","Malaysia",
                 "Maldives","Mali","Malta","Marshall Islands","Mauritania","Mauritius","Mexico",
                 "Micronesia","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique",
                 "Namibia","Nauru","Nepal","Netherlands","New Zealand","Nicaragua","Niger",
                 "Nigeria","North Korea","Norway","Oman","Pakistan","Palau","Palestinian Territories",
                 "PanamaPapua"," New Guinea","Paraguay","Peru","Philippines","Poland","Portugal",
                 "QatarRomania","Russia","Rwanda","Saint Kitts and Nevis","Saint LuciaSaint ",
                 "Vincent and the Grenadines","Samoa","San Marino","Sao Tome and Principe",
                 "Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore",
                 "Sint Maarten","Slovakia","Slovenia","Solomon Islands","Somalia","South Africa",
                 "korea","South Korea","South Sudan","Spain","Sri Lanka","Sudan","Suriname",
                 "Swaziland","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania",
                 "Thailand","Timor-Leste","Togo","Tonga","Trinidad and Tobago","Tunisia",
                 "Turkey","Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates",
                 "United Kingdom","Uruguay","Uzbekistan","Vanuatu","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"]
# $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
tweets_1=pd.read_csv("/home/yousef/twitter_sentiment_corpus.csv",names=["Sender","Sentiment","ID","Date","Content"])
tweets_2=pd.read_table("/home/yousef/training.txt",names = ["Sentiment", "Content"])
tweets_1 = tweets_1[tweets_1.ID != "TweetId"]
tweets_1 = tweets_1[tweets_1.Sentiment != "irrelevant"]
tweets_1=tweets_1.replace(['negative','neutral','positive'],[0,2,1])
tweets_1=tweets_1.drop('Sender',1)
tweets_1=tweets_1.drop('ID',1)
tweets_1=tweets_1.drop('Date',1)
frames = [tweets_1, tweets_2]
tweets_1=pd.concat(frames)
data=tweets_1.Content.values
target=tweets_1.Sentiment.values.astype(np.int)
target=target.reshape(target.shape[0],1)
data=data.reshape(data.shape[0],1)
all_tweets=np.concatenate((data,target),axis=1)
data=all_tweets[:,0]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text_clf_1 = Pipeline([('hash_vectorizer',HashingVectorizer(encoding='string',decode_error='ignore',
                       strip_accents='ascii',analyzer='word',
                       ngram_range=(1,1),norm='l2',non_negative=True)),
                      ('tfidf_transformer',TfidfTransformer()),
                      ('clf', SGDClassifier(loss='epsilon_insensitive',penalty='l2',
                  alpha=1e-3,n_iter=5,random_state=42)),])
print(eval(text_clf_1,10,data,target,method='kfold'))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text_clf_2 = Pipeline([('tfidf_vectorizer',text.TfidfVectorizer(encoding='string',
                        decode_error='ignore',strip_accents='ascii',
                        analyzer='word',ngram_range=(1,1),
                        stop_words=stop_words_list,lowercase=True,
                        min_df=5,max_df=0.8, norm='l2',use_idf=True,sublinear_tf=True)),
                      ('tfidf_transformer',TfidfTransformer()),
                      ('clf', SGDClassifier(loss='epsilon_insensitive',penalty='l2',
                  alpha=1e-3,n_iter=5,random_state=42)),])
print(eval(text_clf_2,10,data,target,method='kfold'))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vectorizer=HashingVectorizer(encoding='string',decode_error='ignore',
                       strip_accents='ascii',analyzer='word',
                       ngram_range=(1,1),norm='l2',non_negative=True)
transformer=TfidfTransformer()
data=vectorizer.fit_transform(data)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mlp=MLPClassifier(hidden_layer_sizes=(50,), solver='sgd',
                  learning_rate='adaptive', max_iter=50, verbose=True)

mlp.fit(X_train,Y_train)
mlp.score(X_test,Y_test)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mlp=MLPClassifier(hidden_layer_sizes=(50,50), solver='sgd',
                  learning_rate='adaptive', max_iter=50, verbose=True)

mlp.fit(X_train,Y_train)
mlp.score(X_test,Y_test)

