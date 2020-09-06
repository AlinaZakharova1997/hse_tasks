
import nltk
from nltk.tokenize import RegexpTokenizer
import csv

# open the text
text = open('dom.txt', 'r').read().splitlines()
# tokenizer object
tokenizer = RegexpTokenizer(r'\w+')
# frequency dictionary for tokens
tokens_dict = {}
# extracting tokens from a given file
for num,line in enumerate(text):
    string = line
    # tokenize every string and make it lower
    for num,token in enumerate(tokenizer.tokenize(string.lower())):
        # make the frequency dictionary
        tokens_dict.setdefault(token, 0)
        tokens_dict[token] += 1
# sort the dictionary        
dict_sort = sorted(tokens_dict.items(),key=lambda i: i[1], reverse = True)
# save the dictionary to a csv file
with open('result.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter= ';')
        for key,value in dict_sort:
            writer.writerow([key, value])
