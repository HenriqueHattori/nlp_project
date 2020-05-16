# Project library
from data_read import *

# External library

data = extract_data()
#print(data)
tokens = tokenize(remove_punctuation(data))
print(tokens)
tokens = remove_stopwords(tokens)
#print(tokens)