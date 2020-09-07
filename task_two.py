import requests
import json
import pymorphy2
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer

# pymorphy2 getting ready
morph=pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
# link and file I want to get
link = 'http://lib.ru/POEZIQ/PESSOA/'
filename = 'lirika.txt'
# getting it by requests and the text is with tags
req = requests.get(link+filename)
text_tags = req.text
# getting rid of tags
soup = BeautifulSoup(text_tags, 'lxml')
text = soup.get_text()
# tokenizing the text
words = tokenizer.tokenize(text.lower())
# getting lemmas by pymorphy2
lemmas = []
for word in words:
    lemma = morph.parse(word)[0].normal_form
    lemmas.append(lemma)
    
# finding lemmas with 2 o letters
lemmas_with_o_letter = []
for lemma in lemmas:
    if 'о' in lemma and lemma.count('о')==2:
        lemmas_with_o_letter.append(lemma)
# writing list in a text.txt file
file = open('o_letters_list.txt','w')
for item in lemmas_with_o_letter:
    file.write("%s\n" % item)
file.close()

# time to create a dictionary!
dictionary = {}
for word in words:
    dictionary.setdefault(word, 0)
    dictionary[word] += 1
# sort the dictionary        
dict_sort = sorted(dictionary.items(),key=lambda i: i[1], reverse = True)
# writing dictionary in a json file
with open('result.json', 'w') as file:
    json.dump(dict_sort, file, ensure_ascii=False)
