import data_read as data
import clusterize
import pandas as pd
import gensim
import numpy as np
import spacy

nlp = spacy.load('pt_core_news_sm')

# dados = data.reviews_comment

def clean_doc (doc):
    text = [token.lemma_ for token in doc if token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    return text

def clean_to_vectors (doc):
    text = [token.vector for token in doc if token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    return text

def send_dados_to_picle(dados):
    dados.to_pickle('dados.pkl')

# uso de spacy
# dados['Docs'] = dados['Content'].apply(lambda x: nlp(x)) # comentarios tokenizados pelo spacy
# dados['Docs_clean'] = dados['Docs'].apply(lambda x: clean_doc(x)) # cada linha sao palavras lematizadas, sem pontuacao e stopwords
# dados['Docs_vector'] = dados['Docs'].apply(lambda x: clean_to_vectors(x)) # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
# send_dados_to_picle(dados)

dados = pd.read_pickle('dados.pkl') #le o dataframe com os resultados do spacy


# dados.to_excel('teste.xlsx')
arr = dados['Docs_vector'].values
# print(dados['Docs_vector'])
to_cluster_vector=[]
# soma todos os vetores palavras do commentario
for comment in dados['Docs_vector']:
    sum_sent_vector=0
    for word_vector in comment:
        sum_sent_vector= sum_sent_vector+word_vector
    to_cluster_vector.append(sum_sent_vector)
t = np.stack(to_cluster_vector,axis=0)
clusterize.clusterize(t)
print('-----WORD-----------------------------------------------------------------------------------')
to_cluster_vector=[]
# soma todos os vetores palavras do commentario
for comment in dados['Docs_vector']:
    for word_vector in comment:
        to_cluster_vector.append(word_vector)

t2 = np.stack(to_cluster_vector,axis=0)
# print(t2)
clusterize.clusterize(t2)
#
