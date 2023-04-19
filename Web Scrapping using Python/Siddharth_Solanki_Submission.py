#!/usr/bin/env python
# coding: utf-8

# In[2]:


#extracting data
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


# Read the input file
df = pd.read_excel('Input.xlsx')

# Loop through each URL in the input file
for i, row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    
    # Fetch the HTML content of the webpage
    response = requests.get(url)
    html = response.content
    
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract the article title and text (excluding p tags with class "tdm-descr" and div tag with id "tdi_20" and class "tdc-zone")
    title_element = soup.find('h1')
    if title_element:
        title = title_element.text.strip()
    else:
        title =' '
    
    article_element = soup.find('div', class_=re.compile('td-post-content.*'))
    if article_element:
        article_elements = article_element.find_all(['p', 'h2', 'h3','h4', 'ul', 'ol', 'blockquote'])
        article =  '\n\n'.join([elem.text.strip() for elem in article_elements if not elem.find_parent('div', {'class': 'wp-block-preformatted'})])
    else:
        article = ' '
        
    print("\n\n", title, "\n\n", article)
    
    # Save the extracted text as a text file
    with open(f'{url_id}.txt', 'w', encoding='utf-8') as f:
        f.write(f'{title}\n\n{article}')


# In[ ]:


import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import cmudict
import string

# Read the Stop Words Lists
stopwords_folder = 'StopWords/'
stopwords_files = ['StopWords_Generic.txt', 'StopWords_Auditor.txt', 'StopWords_Currencies.txt','StopWords_DatesandNumbers.txt','StopWords_GenericLong.txt','StopWords_Geographic.txt','StopWords_Names.txt']

stopwords_list = set()

for file in stopwords_files:
    with open(stopwords_folder+file, 'r') as f:
        words1 = f.read().splitlines()
        stopwords_list.update(words1)
     

#Read the Master Dictionary
folder_name = 'MasterDictionary/'
file_name = 'positive-words.txt'
with open(folder_name + file_name, 'r', encoding='utf-8') as f:
    positive_dict = f.read().splitlines()

file_name = 'negative-words.txt'
with open(folder_name + file_name, 'r', encoding='utf-8') as f:
    negative_dict = f.read().splitlines()
    
##print("Postive: \n", positive_dict,"\n\nNegative: \n", negative_dict)


# Function to calculate the scores
def calculate_scores(text):
    positive_score = 0
    negative_score = 0
    subjectivity_score = 0

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words and convert to lower case
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    

    # Calculate scores for each token
    for token in tokens:
        if token in positive_dict:
            positive_score += 1
            #
            
        if token in negative_dict:
            negative_score -= 1
            ##print(token)
    
    negative_score= negative_score*-1
    
    # Calculate subjectivity score
    subjectivity_score = (positive_score + (negative_score)) / (len(tokens) + 0.000001)

    # Calculate polarity score
    polarity_score = (positive_score - (negative_score)) / ((positive_score + (negative_score)) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score



def count_complex_words(text):
    """
    Returns the number of complex words in a text, defined as words with more than two syllables.
    """

    # Load the CMU pronunciation dictionary
    cmu_dict = nltk.corpus.cmudict.dict()
    
    # Get the set of all words in the text
    words = set(nltk.word_tokenize(text.lower()))
    
    # Count the number of words with more than two syllables
    count = 0
    for word in words:
        # Use the CMU dictionary to get the number of syllables in the word
        syllables = cmu_dict.get(word, [])
        num_syllables = [len(list(y for y in x if y[-1].isdigit())) for x in syllables]
        if num_syllables and max(num_syllables) > 2:
            count += 1
            ##print(word)
    
    return count



def calculate_readability(text):
    """
    Calculates the readability analysis using the Gunning Fox index formula.
    Returns a tuple containing the average sentence length, percentage of complex words, and the fog index.
    """
    sentences = nltk.sent_tokenize(text)
    # Tokenize the text into sentences and words
    num_sentences = len(sentences)
    
    ##print ("num_sentences: ", num_sentences)
    
    # Calculate the total number of words and the total number of words in each sentence
    total_words = 0
    words_per_sentence = []
    num_words=0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        num_words = len(words)
        total_words += num_words
        words_per_sentence.append(num_words)
   ##print("words_per_sentence: ", words_per_sentence)
    
    # Calculate the average sentence length and average number of words per sentence
    if num_sentences > 0:
        avg_sentence_length = sum(words_per_sentence) / num_sentences
        avg_words_per_sentence = total_words / num_sentences
        # Count the number of complex words
        percentage_complex_words = complex_words / num_words
    
        # Calculate the fog index
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        ##print(total_words, sum(words_per_sentence))
    else:
        avg_sentence_length = 0
        avg_words_per_sentence = 0
        percentage_complex_words=0
        fog_index=0
        
    return (avg_sentence_length, avg_words_per_sentence, percentage_complex_words, fog_index)


# Calculate the number of words cleaned
def count_cleaned_words(text):
    """
    Counts the total number of cleaned words in the text by removing stopwords and punctuations.
    """
    # Load the stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuations and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Return the count of cleaned words
    return len(cleaned_words)

#Calculate number of syllables
def count_syllables(word):
    """
    Returns the number of syllables in a given word.
    """
    # remove any ending punctuation
    word = re.sub(r'[^\w\s]', '', word)
    
    # handle exceptions for words ending in "es" or "ed"
    if word.endswith(('es', 'ed')):
        word = word[:-2]
    
    # count the number of vowels
    num_vowels = len(re.findall(r'[aeiouy]+', word, re.IGNORECASE))
    
    # handle special cases
    if word.endswith('e'):
        num_vowels -= 1
    if re.search(r'[aeiouy]{3}', word, re.IGNORECASE):
        num_vowels += 1
    if num_vowels == 0:
        num_vowels = 1
    
    return num_vowels

def syllable_per_word(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    total_syllable=0
    total_words=len(words)
    
    if(total_words>0):
        
        for word in words:
            num=count_syllables(word)
            total_syllable+=num
        return total_syllable/total_words   
    else:
        return 0



#Returns the count of personal pronouns in a text, excluding the country name 'US'.
def count_personal_pronouns(text):

    personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
    # regex pattern to find personal pronouns
    pattern = r'\b(?:{})\b'.format('|'.join(personal_pronouns))
    # find all matches of personal pronouns
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # filter out matches of 'US' country name
    matches = [match for match in matches if match.lower()!= 'us']
    ##print( matches)
    # return the count of personal pronouns
    return len(matches)

#Returns the average word length in a given text.
def average_word_length(text):

    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Calculate the total number of characters in all words
    total_chars = sum(len(word) for word in words)
    
    # Calculate the total number of words
    total_words = len(words)
    #print("total words+ ", total_words)
    
    if total_words>0: 
        # Calculate the average word length
        avg_word_length = total_chars / total_words
    else:
        avg_word_length=0
    
    return avg_word_length


# Read the input file
df = pd.read_excel('Input.xlsx')


url_id = 0
url = 0
positive_score = 0
negative_score = 0
polarity_score = 0
subjectivity_score = 0
avg_sentence_length = 0
percentage_complex_words = 0
fog_index = 0
avg_words_per_sentence = 0
complex_words = 0
word_count = 0
num_syllables = 0
num_pronouns = 0
avg_word_length = 0


Output_Data_Structure= {"URL_ID": [], "URL": [] , "POSITIVE SCORE":[], "NEGATIVE SCORE": [],
                        "POLARITY SCORE": [], "SUBJECTIVITY SCORE":[], "AVG SENTENCE LENGTH": [],
                        "PERCENTAGE OF COMPLEX WORDS":[] , "FOG INDEX":[],
                        "AVG NUMBER OF WORDS PER SENTENCE":[] , "COMPLEX WORD COUNT":[] , "WORD COUNT":[], "SYLLABLE PER WORD":[] ,
                        "PERSONAL PRONOUNS": [] , "AVG WORD LENGTH": [] 
                       }

# Loop through each URL in the input file
for i, row in df.iterrows():
    url_id = row['URL_ID']
    url=row['URL']
    
    # Read the extracted text file
    with open(f'{url_id}.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    words = nltk.word_tokenize(text)

    # Clean the text by removing non-alphabetic characters
    clean_text = re.sub('[^A-Za-z]+', ' ', text)
    
    

    # Calculate the scores
    positive_score, negative_score, polarity_score, subjectivity_score = calculate_scores(clean_text)
    complex_words=count_complex_words(text)
    avg_sentence_length, avg_words_per_sentence, percentage_complex_words, fog_index= calculate_readability(text)
    word_count=count_cleaned_words(text)
    num_syllables=syllable_per_word(clean_text)
    num_pronouns= count_personal_pronouns(text)
    Average_Word_Length= average_word_length(clean_text)

#   # #print the scores
 #   print(f'\nURL ID: {url_id}')
#     #print(f'Positive Score: {positive_score}')
#     #print(f'Negative Score: {negative_score}')
#     #print(f'Polarity Score: {polarity_score}')
#     #print(f'Subjectivity Score: {subjectivity_score}')
#     #print('Complex words:', complex_words)   
#     #print(f'Average Sentence Length: {avg_sentence_length}')
#     #print(f'Percentage of Complex Words: {percentage_complex_words}')
#     #print(f'Fog Index: {fog_index}')  
#     #print(f'Average words per sentence: {round(avg_words_per_sentence)} words')
#    print(f'Total number of words: {word_count}')
#     #print(f'Number of syllables per word: {num_syllables}')
#     #print(f'Number of Personal Pronouns: {num_pronouns}')
#     #print(f'Average word length: {Average_Word_Length}')
#     #print(text)
    
    Output_Data_Structure["URL_ID"].append(url_id)
    Output_Data_Structure["URL"].append(url)
    Output_Data_Structure['AVG NUMBER OF WORDS PER SENTENCE'].append(avg_words_per_sentence)
    Output_Data_Structure['AVG SENTENCE LENGTH'].append(avg_sentence_length)
    Output_Data_Structure['AVG WORD LENGTH'].append(Average_Word_Length)
    Output_Data_Structure['COMPLEX WORD COUNT'].append(complex_words)
    Output_Data_Structure['FOG INDEX'].append(fog_index)
    Output_Data_Structure['NEGATIVE SCORE'].append(negative_score)
    Output_Data_Structure['PERCENTAGE OF COMPLEX WORDS'].append(percentage_complex_words)
    Output_Data_Structure['PERSONAL PRONOUNS'].append(num_pronouns)
    Output_Data_Structure['POLARITY SCORE'].append(polarity_score)
    Output_Data_Structure['POSITIVE SCORE'].append(positive_score)
    Output_Data_Structure['SUBJECTIVITY SCORE'].append(subjectivity_score)
    Output_Data_Structure['SYLLABLE PER WORD'].append(num_syllables)
    Output_Data_Structure['WORD COUNT'].append(word_count)
    
    
    

#print(Output_Data_Structure)
    
# # convert the dictionary to a pandas dataframe
df = pd.DataFrame(Output_Data_Structure)

# # save the dataframe to an Excel file
df.to_excel('Final Output Data Structure.xlsx', index=False)


# In[ ]:




