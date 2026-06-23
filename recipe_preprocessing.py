import pandas as pd
import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

REPLACE_BY_SPACE_RE = re.compile(r'[\-\(\)\{\}\[\]\/|@,;]')
# BAD_SYMBOLS_RE = re.compile(r'[^a-z0-9_À-ÿ #+_]')
BAD_SYMBOLS_RE = re.compile('[^a-zÀ-ÿ ]')
# FRACTIONS_RE = re.compile('[^¼⅓½⅔¾]')
STOPWORDS = set(stopwords.words('english'))
MEASUREMENTS = re.compile(r'(teaspoons?)|( table[a-z]*)|(cups?)|(ounces?)|(pounds?)|(grams?)|(chopped)|(minced)|(fresh)|(min[a-z]*)')

def clean_text(text):
  wnl = WordNetLemmatizer()
  text = text.lower()
  text = REPLACE_BY_SPACE_RE.sub(' ', text)
  text = BAD_SYMBOLS_RE.sub('', text)
  text = MEASUREMENTS.sub('',text)
  # text = 
  text = ' '.join(wnl.lemmatize(word) for word in text.split() if word not in STOPWORDS) # delete stopwors from text
  return text

def preprocess_recipe(name, ingredients, instructions):     
    cleaned_text = '<T> ' + clean_text(name) + ' <I> ' + clean_text(ingredients) + ' <D> ' + clean_text(instructions)
    return cleaned_text