import pandas as pd
cui='C0020538'
df=pd.read_csv('DiseaseUMLS.csv')
def generate_concept(cui):
    return df.UMLS[df.CUI==cui].str()
if __name__=='__main__':
    print(generate_concept(cui))