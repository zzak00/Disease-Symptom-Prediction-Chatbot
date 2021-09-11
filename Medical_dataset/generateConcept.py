import pandas as pd
import sys

df=pd.read_csv('DiseaseUMLS.csv')
def generate_concept(cui):
    return df.UMLS[df.CUI==cui].values[0]


if __name__=='__main__':
    print(generate_concept(sys.argv[1]))