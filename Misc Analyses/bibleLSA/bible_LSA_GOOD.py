
# coding: utf-8

# In[22]:

# imports
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD



# In[23]:

# downloading stopwords should they not be present
nltk.download('stopwords')


# In[24]:


raw_bible = open('bibledata.dat', 'r').read()


# In[25]:

# In XML, the raw text content is arranged and separated by tagging. Below I will extract only data from text tags
# First, instantiate beautiful 'soup' with params (data_file,'file type') Note var name must be soup!
soup = BeautifulSoup(raw_bible, 'lxml')


# In[26]:

# Filter the soup for only the values found between the <text> tags, rename the variable for ease of reading.
raw1 = soup.findAll('v')
#print(raw1)

# Generate the corpus with some list comprehension, iterates for all documents all text and appends docs to postDocs.
# Also, now the data will no longer be just text, so new var name postDocs 
verses = [x.text for x in raw1]


# In[27]:

len(verses)


# In[28]:

verses = [x.lower() for x in verses]


# In[29]:

len(verses)


# In[30]:

len(verses)


# In[ ]:

verses2 = list()
i = 0
while i < len(verses):
    verses2.append(verses[i] + verses[i + 1])
    if (i > 0):
        i = i + 2


# In[ ]:

len(verses2)


# In[33]:

verses3 = list()
i = 0
while i < len(verses2):
    verses3.append(verses2[i] + verses2[i + 1])
    i = i + 2
    


# In[34]:

len(verses3)


# In[17]:

verses4 = list()
i = 0
while i < len(verses3):
    verses4.append(verses3[i] + verses3[i + 1] + verses3[i + 2])
    i = i + 3


# In[18]:

len(verses4)


# In[ ]:

verses5 = list()
i = 0
while i < len(verses4):
    verses5.append(verses4[i] + verses4[i + 1] + verses4[i + 2])
    i = i + 3


# In[ ]:

stopset = set(stopwords.words('english'))
stopset.update(['&#x27;s','x27','/',',','.','#','lt','p','/p','br','amp','quot','field','font','normal','span','0px','rgb','style','51', 
                'spacing','text','helvetica','size','family', 'space', 'arial', 'height', 'indent', 'letter'
                'line','none','sans','serif','transform','line','variant','weight','times', 'new','strong', 'video',
                'title','white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class',  ])


# In[ ]:

# Import the Terrier stopset from file, union with existing stopset
terrierstopset = open('terrierstopset.txt', 'r').read()
stopset = set(stopset).union(set(terrierstopset))


# In[16]:

# Before...
print(len(verses4))


# In[ ]:

# Define the vectorizer model
vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(2, 7))

# Fit the corpus data
X = vectorizer.fit_transform(verses4)


# In[ ]:

# Tada! This is now the output of the first document in the corpus, in sparse IDF matrix form.
print(X[0])


# In[ ]:

# The current shape is (documents, terms)
X.shape


# In[ ]:

# Begin by defining the TruncatedSVD model (num rows/docs?, how many passes over the data (epochs)? )
#Note: n_iter defaults to 5 if not passed, and 1 if using partial_fit
lsa = TruncatedSVD(n_components=100, n_iter=5)

# Fit the model
lsa.fit(X)


# In[ ]:

import sys
print (sys.version)


# In[ ]:

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


# In[ ]:



