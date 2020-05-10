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

def clean_do_vectors (doc):
    text = [token.vector for token in doc if token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    return text

def send_dados_to_picle(dados):
    dados.to_pickle('dados.pkl')

# uso de spacy
# dados['Docs'] = dados['Content'].apply(lambda x: nlp(x)) # comentarios tokenizados pelo spacy
# dados['Docs_clean'] = dados['Docs'].apply(lambda x: clean_doc(x)) # cada linha sao palavras lematizadas, sem pontuacao e stopwords
# dados['Docs_vector'] = dados['Docs'].apply(lambda x: clean_do_vectors(x)) # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
# send_dados_to_picle(dados)

dados = pd.read_pickle('dados.pkl')

dados['Docs_sum_vector'] = dados['Docs_vector'].apply(lambda x: np.sum(x,axis=1))

array_vector = np.array(dados['Docs_sum_vector'].reset_index(drop=True).to_numpy)
arr = dados['Docs_sum_vector'].values
print(type(arr))
print(arr)
clusterize.clusterize(arr)

# dictionary = gensim.corpora.Dictionary(dados['Clean_Content_Vector'])
# doc_term_matrix = [dictionary.doc2bow(rev) for rev in dados['Clean_Content_Vector']]
# print(dictionary.id2token)
