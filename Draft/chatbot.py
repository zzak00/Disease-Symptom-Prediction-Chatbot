import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import joblib
#knn_from_joblib.predict(X_test) 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

with open('../Medical_dataset/intents_short.json', 'r') as f:
    intents = json.load(f)
    
intents
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()
knn= joblib.load('../model/knn.pkl')  

# preprocess sentence
def preprocess_sent(sent):
    t=nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t if (w not in set(stopwords.words('english')) and w.isalpha())])

# BOW of prepocessed sentence
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# predict possible symptom in a sentence
def predictSym(sym,vocab,app_tag):
    sym=preprocess_sent(sym)
    bow=np.array(bag_of_words(sym,vocab))
    res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
    order=np.argsort(res)[::-1].tolist()
    possym=[]
    for i in order:
        if app_tag[i].replace('_',' ') in sym:
            return app_tag[i],1
        if app_tag[i] not in possym and res[i]!=0:
            possym.append(app_tag[i])
    return possym,0

# input : patient symptoms / output : OHV DataFrame 
def OHV(cl_sym,all_sym):
    l=np.zeros([1,len(all_sym)])
    for sym in cl_sym:
        l[0,all_sym.index(sym)]=1
    return pd.DataFrame(l, columns =all_symp)

def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

# returns possible diseases 
def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis

# input: Disease / output: all symptoms
def symVONdisease(df,disease):
    ddf=df[df.prognosis==disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# preprocess symptoms    
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')     

def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('../Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('../Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('../Medical_dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")

getSeverityDict()
getprecautionDict()
getDescription()
# read TF IDF symptoms and Training diseases
df=pd.read_csv('../Medical_dataset/tfidfsymptoms.csv')
df_tr=pd.read_csv('../Medical_dataset/Training.csv')
vocab=list(df.columns)
disease=df_tr.iloc[:,-1].to_list()
all_symp_col=list(df_tr.columns[:-1])
all_symp=[clean_symp(sym) for sym in (all_symp_col)]
app_tag=[]
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        app_tag.append(tag)


def main_sp(name):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Hi Mr/Ms "+name+", can you describe you main symptom ?  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    psym1,find=predictSym(sym1,vocab,app_tag)
    if find==1:
        sym1=psym1
    else:
        i=0
        while True and i<len(psym1):
            print('Do you experience '+psym1[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym1=psym1[i]
                break
            else:
                i=i+1

    print("Is there any other symtom Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    psym2,find=predictSym(sym2,vocab,app_tag)
    if find==1:
        sym2=psym2
    else:
        i=0
        while True and i<len(psym2):
            print('Do you experience '+psym2[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym2=psym2[i]
                break
            else:
                i=i+1
    
    #create patient symp list
    all_sym=[sym1,sym2]
    #predict possible diseases
    diseases=possible_diseases(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in diseases:
        if stop==False:
            for sym in symVONdisease(df_tr,dis):
                if sym not in all_sym:
                    print(clean_symp(sym)+' ?')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if inp=="yes":
                        all_sym.append(sym)
                        dise=possible_diseases(all_sym)
                        if len(dise)==1:
                            stop=True 
                            break
                    else:
                        continue
    return knn.predict(OHV(all_sym,all_symp_col)),all_sym

def chat_sp():
    a=True
    while a:
        name=getInfo()
        result,sym=main_sp(name)
        if result == None :
            ans3=input("can you specify more what you feel or tap q to stop the conversation")
            if ans3=="q":
                a=False
            else:
                continue

        else:
            print("you may have "+result[0])
            print(description_list[result[0]])
            an=input("how many day do you feel those symptoms ?")
            if calc_condition(sym,int(an))==1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precautionDictionary[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                a=False
                print("!!!!! thanks for using ower application !!!!!! ")

if __name__=='__main__':
    chat_sp()