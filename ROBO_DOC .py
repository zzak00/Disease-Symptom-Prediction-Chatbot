import pandas as pd
import numpy as np
from nltk.corpus import wordnet 
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib

nlp = spacy.load('en_core_web_sm')

#save data
data={"users":[]}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)


def write_json(new_data, filename='DATA.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)
 
    # python object to be appended
y = {"Name":"Nikhil",
     "Age": "nikhil@geeksforgeeks.org",
     "Gender": "Full Time",
     "Disease":"",
     "Sympts":""
    }
     
write_json(y)

df_tr=pd.read_csv('Medical_dataset/Training.csv')
df_tt=pd.read_csv('Medical_dataset/Testing.csv')

symp=[]
disease=[]
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i]==1].to_list())
    disease.append(df_tr.iloc[i,-1])


# # I- GET ALL SYMPTOMS

all_symp_col=list(df_tr.columns[:-1])
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow') 

all_symp=[clean_symp(sym) for sym in (all_symp_col)]

def preprocess(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)

all_symp_pr=[preprocess(sym) for sym in all_symp]

#associate each processed symp with column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# II- Syntactic Similarity

#Returns all the subsets of a set. This is a generator.
#{1,2,3}->[{},{1},{2},{3},{1,3},{1,2},..]
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


#Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if len(a[j])>len(a[i]):
                a[i],a[j]=a[j],a[i]
    a.pop()
    return a

# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return([' '.join(permutation) for permutation in permutations])

# check if a txt and all diferrent combination if it exists in processed symp list
def DoesExist(txt):
    txt=txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations :
        #print(permutations(comb))
        for sym in permutations(comb):
            if sym in all_symp_pr:
                #print(sym)
                return sym
    return False

#Jaccard similarity 2docs
def jaccard_set(str1, str2):
    list1=str1.split(' ')
    list2=str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# apply vanilla jaccard to symp with all corpus
def syntactic_similarity(symp_t,corpus):
    most_sim=[]
    poss_sym=[]
    for symp in corpus:
        d=jaccard_set(symp_t,symp)
        most_sim.append(d)
    order=np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(corpus[i]) :
            return 1,[corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i]!=0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1,poss_sym
    else: return 0,None


# check a pattern if it exists in processed symp list
def check_pattern(inp,dis_list):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,None


# III- Semantic Similarity


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
def WSD(word, context):
    sens=lesk(context, word)
    return sens

# semantic similarity 2docs
def semanticD(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess(doc2).split(' ')
    score=0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1,doc1)
            syn2 = WSD(tock2,doc2)
            if syn1 is not None and syn2 is not None :
                x=syn1.wup_similarity(syn2)
                #x=syn1.path_similarity((syn2))
                if x is not None and x>0.25:
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))

#apply semantic simarity to symp with all corpus
def semantic_similarity(symp_t,corpus):
    max_sim=0
    most_sim=None
    for symp in corpus:
        d=semanticD(symp_t,symp)
        if d>max_sim:
            most_sim=symp
            max_sim=d
    return max_sim,most_sim


# given a symp suggest possible synonyms 
def suggest_syn(sym):
    symp=[]
    synonyms = wordnet.synsets(sym)
    lemmas=[word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res,sym1=semantic_similarity(e,all_symp_pr)
        if res!=0:
            symp.append(sym1)
    return list(set(symp))



# One-Hot-Vector dataframe
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

# list of symptoms --> possible diseases
def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis


# disease --> all symptoms
def symVONdisease(df,disease):
    ddf=df[df.prognosis==disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()
    


# IV- Prediction Model (KNN)
#load model 
knn_clf=joblib.load('model/knn.pkl')  


# ##  VI- SEVERITY / DESCRIPTION / PRECAUTION
# get dictionaries for severity-description-precaution for all diseases

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

#load dictionaries
getSeverityDict()
getprecautionDict()
getDescription()



# calcul patient condition
def calc_condition(exp,days):
    sum=0
    for item in exp:
        if item in severityDictionary.keys():
            sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")


#  Chat



def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)


def related_sym(psym1):
    if len(psym1)==1:
        return psym1[0]
    print("searches related to input: ")
    for num,it in enumerate(psym1):
        print(num,")",clean_symp(it))
    if num!=0:
        print(f"Select the one you meant (0 - {num}):  ", end="")
        conf_inp = int(input(""))
    else:
        conf_inp=0

    disease_input=psym1[conf_inp]
    return disease_input


def main_sp(name,all_symp_col):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Enter the main symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    sym1=preprocess(sym1)
    sim1,psym1=syntactic_similarity(sym1,all_symp_pr)
    if sim1==1:
        psym1=related_sym(psym1)
    
    #get the 2nd syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)

    print("Enter a second symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    sym2=preprocess(sym2)
    sim2,psym2=syntactic_similarity(sym2,all_symp_pr)
    if sim2==1:
        psym2=related_sym(psym2)
        
    #if check_pattern==0 no similar syntaxic symp1 or symp2 ->> try semantic similarity
    
    if sim1==0 or sim2==0:
        sim1,psym1=semantic_similarity(sym1,all_symp_pr)
        sim2,psym2=semantic_similarity(sym2,all_symp_pr)
        
        #if semantic sim syp1 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim1==0:
            sugg=suggest_syn(sym1)
            print('Are you experiencing any ')
            for res in sugg:
                print(res)
                inp=input('')
                if inp=="yes":
                    psym1=res
                    sim1=1
                    break
                
        #if semantic sim syp2 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim2==0:
            sugg=suggest_syn(sym2)
            for res in sugg:
                inp=input('Do you feel '+ res+" ?(yes or no) ")
                if inp=="yes":
                    psym2=res
                    sim2=1
                    break
        #if no syntaxic semantic and suggested sym found return None and ask for clarification

        if sim1==0 and sim2==0:
            return None,None
        else:
            # if at least one sym found ->> duplicate it and proceed
            if sim1==0:
                psym1=psym2
            if sim2==0:
                psym2=psym1
    #create patient symp list
    all_sym=[col_dict[psym1],col_dict[psym2]]
    #predict possible diseases
    diseases=possible_diseases(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in diseases:
        print(diseases)
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
                        diseases=possible_diseases(all_sym)
                        if len(diseases)==1:
                            stop=True 
    return knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym

    
    


# In[90]:


def chat_sp():
    a=True
    while a:
        name=getInfo()
        result,sym=main_sp(name,all_symp_col)
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




