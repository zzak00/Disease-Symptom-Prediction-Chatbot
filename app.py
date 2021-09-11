import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn 
import csv
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import wordnet
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request,session

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
j=0
#read pd
df_tr=pd.read_csv('NEWTRAIN.csv')
df_tt=pd.read_csv('NEWTEST.csv')
symp=[]
disease=[]
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i]==1].to_list())
    disease.append(df_tr.iloc[i,-1])



# clean symp
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow') 

all_symp_col=list(df_tr.columns[:-1])
all_symp=[clean_symp(sym) for sym in (all_symp_col)]

# process Text
# User Text
def preprocess(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha() and token.tag_ in ("JJ","NN","VBG","NNS")):
            d.append(token.lemma_.lower() )
    return ' '.join(d)

# Sympt Text
def preprocess_sym(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)

all_symp_pr=[preprocess_sym(sym) for sym in all_symp]

#associe chaque symp pretraite au non de sa colonne originale
col_dict = dict(zip(all_symp_pr, all_symp_col))

# syntaxic similrity
# Jaccard
def jaccard_set(str1, str2):
    list1=str1.split(' ')
    list2=str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# Jaccard --> Corpus
def syntactic_similarity(symp_t,corpus):
    max_sim=0
    most_sim=None
    for symp in corpus:
        d=jaccard_set(symp_t,symp)
        if d>max_sim:
            most_sim=symp
            max_sim=d
    return max_sim,most_sim


# Regular expression check
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


# Semantic Similarity
# WSD using LESK
def WSD(word, context):
    sens=lesk(context, word)
    return sens

# Semantic Distance
def semanticD(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess_sym(doc2).split(' ')
    score=0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1,doc1)
            syn2 = WSD(tock2,doc2)
            if syn1 is not None and syn2 is not None :
                x=syn1.path_similarity(syn2)
                if x is not None and x>0.25:
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))

# Sematic Similarity --> Corpus
def semantic_similarity(symp_t,corpus):
    max_sim=0
    most_sim=None
    for symp in corpus:
        d=semanticD(symp_t,symp)
        if d>max_sim:
            most_sim=symp
            max_sim=d
    return max_sim,most_sim

# Suggest Sympts
def suggest_syn(sym):
    symp=[]
    synonyms = wordnet.synsets(sym)
    lemmas=[word.lemma_names() for word in synonyms]
    lemmas = list(set(chain(*lemmas)))
    for e in lemmas:
        res,sym1=semantic_similarity(e,all_symp_pr)
        if res!=0:
            symp.append(sym1)
    return list(set(symp))

# OHV vectorization
#recoit client_symptoms et renvoit un dataframe avec 1 pour les symptoms associees
def OHV(cl_sym,all_sym):
    l=np.zeros([1,len(all_sym)])
    for sym in cl_sym:
        l[0,all_sym.index(sym)]=1
    return pd.DataFrame(l, columns =all_sym)

# get all Possible diseases

def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

#recoit une maladie renvoit tous les sympts
def symVONdisease(df,disease):
    ddf=df[df.prognosis==disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()
    

def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis


# Prediction Model 
X_train=df_tr.iloc[:,:-1]
y_train = df_tr.iloc[:,-1] 
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

## SEVERITY / DESCRIPTION / PRECAUTION
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

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
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def calc_condition(exp,days):
    sum=0
    for item in exp:
        if item in severityDictionary:
            sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        print("You should take the consultation from doctor. ")
        return 1
    else:
        print("It might not be that bad but you should take precautions.")
        return 0


getSeverityDict()
getprecautionDict()
getDescription()


# Chat 
def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)

def related_sym(psym1):
    s="searches related to input: <br>"
    i=len(s)
    for num,it in enumerate(psym1):
        s+=str(num)+") "+clean_symp(it)+"<br>"
    if num!=0:
        s+="Select the one you meant."
        return s
    else:
        return 0

    disease_input=psym1[conf_inp]
    return disease_input

def main_sp(name,all_symp_col):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Enter the main symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    sym1=preprocess_sym(sym1)
    sim1,psym1=check_pattern(sym1,all_symp_pr)
    if sim1==1 :
        psym1=related_sym(psym1)
    
    #get the 2nd syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)

    print("Enter a second symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    sym2=preprocess_sym(sym2)
    sim2,psym2=check_pattern(sym2,all_symp_pr)
    if sim2==1 :
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
    return knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym

"""
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
                print("§ Thanks for using ower application § ")"""
import json
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

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/get")
def get_bot_response():
    s = request.args.get('msg')
    if "step" in session:
        if session["step"]=="Q_C":
            name=session["name"]
            age=session["age"]
            gender=session["gender"]
            session.clear()
            if s=="q":
                "Thank you for using ower web site Mr/Ms "+name
            else:
                session["step"]="FS"
                session["name"]=name  
                session["age"]=age
                session["gender"]=gender  
    if 'name' not in session and 'step' not in session:
        session['name']=s
        session['step']="age"
        return "please give us your age "
    if session["step"]=="age":
        session["age"]=int(s)
        session["step"]="gender"
        return "please give us your gender"
    if session["step"]=="gender":
        session["gender"]=s
        session["step"]="Depart"
    if session['step']=="Depart":
        session['step']="FS" #first symptom
        return "HeLLO Mr/Ms "+session["name"]+" enter the main symptom you are experiencing "
    if session['step']=="FS":
        sym1 = s
        sym1=preprocess_sym(sym1)
        sim1,psym1=check_pattern(sym1,all_symp_pr) 
        temp=[]
        temp.append(sym1)
        temp.append(sim1)
        temp.append(psym1)
        session['FSY']=temp #info du 1er symptome
        session['step']="SS" #second symptomee     
        if sim1==1:
            session['step']="RS1" #related_sym1
            s=related_sym(psym1)
            if s!=0:
                return s
        else:
            return "Enter a second symptom you are experiencing Mr/Ms "+session["name"]
    if session['step']=="RS1":
        temp=session['FSY']
        psym1=temp[2]
        psym1=psym1[int(s)]
        temp[2]=psym1
        session['FSY']=temp
        session['step']='SS'
        return "Enter a second symptom you are experiencing Mr/Ms "+session["name"]
    if session['step']=="SS":
        sym2 = s
        sym2=preprocess_sym(sym2)
        sim2,psym2=check_pattern(sym2,all_symp_pr) 
        temp=[]
        temp.append(sym2)
        temp.append(sim2)
        temp.append(psym2)
        session['SSY']=temp #info du 2eME symptome(sym,sim,psym)
        session['step']="semantic" #face semantic
        if sim2==1:
            session['step']="RS2" #related sym2            
            s=related_sym(psym2)
            if s!=0:
                return s
    if session['step']=="RS2":
        temp=session['SSY']
        psym2=temp[2]
        psym2=psym2[int(s)]
        temp[2]=psym2
        session['SSY']=temp
        session['step']="semantic"
    if session['step']=="semantic":
        temp=session["FSY"] #recuperer info du premier
        sym1=temp[0]
        sim1=temp[1]
        temp=session["SSY"] #recuperer info du 2 eme symptome
        sym2=temp[0]
        sim2=temp[1]
        if sim1==0 or sim2==0:
            sim1,psym1=semantic_similarity(sym1,all_symp_pr)
            sim2,psym2=semantic_similarity(sym2,all_symp_pr)
            temp=[]
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['SSY']=temp
            temp=[]
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['FSY']=temp
            session['step']="sim1=0"
        else:
            print("hey1")
            session['step']='PD' #to possible_diseases
    if session['step']=="sim1=0": #test syntaxic
        print("innnn")
        temp=session["FSY"]
        sym1=temp[0]
        sim1=temp[1]
        if sim1==0:
            if "suggested" in session :
                sugg=session["suggested"]
                if s=="yes":
                    psym1=sugg[0]
                    sim1=1
                    temp=session["FSY"]
                    temp[1]=sim1
                    temp[2]=psym1
                    session["FSY"]=temp
                    sugg=[]
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"]=suggest_syn(sym1)
                sugg=session["suggested"]
            if len(sugg)>0:
                msg="Do you feel "+sugg[0]+"?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step']="sim2=0"
    if session['step']=="sim2=0":
        temp=session["SSY"]
        sym2=temp[0]
        sim2=temp[1]        
        if sim2==0:
            if "suggested_2" in session :
                sugg=session["suggested_2"]
                if s=="yes":
                    psym2=sugg[0]
                    sim2=1
                    temp=session["SSY"]
                    temp[1]=sim2
                    temp[2]=psym2
                    session["SSY"]=temp
                    sugg=[]
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"]=suggest_syn(sym2)
                sugg=session["suggested_2"]
            if len(sugg)>0:
                msg="Do you feel "+sugg[0]+"?"
                session["suggested_2"]=sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step']="TEST" #test if semantic and syntaxic not found
    if session['step']=="TEST":
        temp=session["FSY"]
        sim1=temp[1]
        psym1=temp[2]
        temp=session["SSY"]
        sim2=temp[1]
        psym2=temp[2]
        if sim1==0 and sim2==0:
            #GO TO THE END
            result=None
            session['step']="END"
        else :
            if sim1==0:
                psym1=psym2
                temp=session["FSY"]
                temp[2]=psym2
                session["FSY"]=temp
            if sim2==0:
                psym2=psym1
                temp=session["SSY"]
                temp[2]=psym1
                session["SSY"]=temp
            session['step']='PD' #to possible_diseases
    if session['step']=='PD': 
        #MAYBE THE LAST STEP
        #create patient symp list
        temp=session["FSY"]
        sim1=temp[1]
        psym1=temp[2]
        temp=session["SSY"]
        sim2=temp[1]
        psym2=temp[2]
        print("hey2")
        if "all" not in session:
            print("inside")
            session["all"]=[col_dict[psym1],col_dict[psym2]]
            print(session["all"])
        session["diseases"]=possible_diseases(session["all"])
        print("hey3")
        print(session["diseases"])
        all_sym=session["all"]
        if len(session["diseases"])<=1:
            session['step']="PREDICT"
        else:
            diseases=session["diseases"]
            dis=diseases[0]
            session["dis"]=dis
            session['step']="for_dis"
    if session['step']=="DIS":
        if "symv" in session:
            if len(s)>0:
                symts=session["symv"]
                all_sym=session["all"]
                if s=="yes":
                    all_sym.append(symts[0])
                    session["all"]=all_sym
                del symts[0]
                session["symv"]=symts
        if "symv" not in session :
            session["symv"]=symVONdisease(df_tr, session["dis"])
        if len(session["symv"])>0:
            if symts[0] not in session["all"]:
                symts=session["symv"]
                msg="do you feel "+clean_symp(symts[0])+"?"
                return msg
            else :
                del symts[0]
                session["symv"]=symts
                return get_bot_response()
        else:
            diseases=session["diseases"]
            del diseases[0]
            session["diseases"]=diseases
            session['step']="for_dis"
    if session['step']=="for_dis":
        diseases=session["diseases"]
        if len(diseases)<0 or len(possible_diseases(session["all"]))<=1:
            session['step']='PREDICT'
        else:
            session["dis"]=diseases[0]
            session['step']="DIS"
            session["symv"]=symVONdisease(df_tr, session["dis"])
            return get_bot_response() #turn around sympt of dis    
        #predict possible diseases 
    if session['step']=="PREDICT":
        result=knn_clf.predict(OHV(session["all"],all_symp_col))
        session['step']="END"
    if session['step']=="END":
        if result!=None:
            session['step']="Description"
            session["disease"]=result[0]
            return "You may have "+result[0]+" type any key to get a description of the disease ."
        else:
            session['step']="Q_C" #test if user want to continue the conversation or not
            return "can you specify more what you feel or type q to stop the conversation"
    if session['step']=="Description":
        y = {"Name":session["name"],"Age": session["age"],"Gender": session["gender"],"Disease":session["disease"],"Sympts":session["all"]}
        write_json(y)
        session['step']="Severity"
        if session["disease"] in description_list.keys():
            return description_list[session["disease"]]+" \n <br> how many day do you feel those symptoms ?"
        else:
            if " " in session["disease"]:
                session["disease"]=session["disease"].replace(" ","_")
            return "please visit <a href='" + "https://en.wikipedia.org/wiki/" +session["disease"]+ "'>  here  </a>"
    if session['step']=="Severity":
        session['step']='FINAL'
        if calc_condition(session["all"],int(s))==1:
            return "you should take the consultation from doctor <br> (type q to end)"
        else:
            msg='Take following precautions :<br> ' 
            i=1
            for e in precautionDictionary[session["disease"]]:
                msg+='\n '+str(i)+'->'+e+'<br>'
            msg+=' (Type q to end)'
            return msg
    if session['step']=="FINAL":
        session['step']="BYE"
        return "do you need another medical consultation (yes or no)? "
    if session['step']=="BYE":
        name=session["name"]
        session.clear()
        if s =="yes":
            session["name"]=name
            session['step']="FS"
            return "HeLLO again Mr/Ms "+session["name"]+" enter the main symptom you are experiencing "
        else:
            return "THANKS Mr/Ms "+name+" for using ower app for more information pleas contact <b> +21266666666</b>"


if __name__ == "__main__":
    import random # define the random module 
    import string  
    S = 10  # number of characters in the string.  
    # call random.choices() string module to find the string in Uppercase + numeric data.  
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S))    
    #chat_sp()
    app.secret_key = str(ran)   
    app.run()