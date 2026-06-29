import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

REPLACE_BY_SPACE_RE = re.compile(r'[\-\(\)\{\}\[\]\/|@,;]')
BAD_SYMBOLS_RE = re.compile('[^a-zÀ-ÿ ]')
STOPWORDS = set(stopwords.words('english'))
MEASUREMENTS = re.compile(r'(teaspoons?)|( table[a-z]*)|(cups?)|(ounces?)|(pounds?)|(grams?)|(chopped)|(minced)|(fresh)|(min[a-z]*)')

def clean_text(text):
  wnl = WordNetLemmatizer()
  text = text.lower()
  text = REPLACE_BY_SPACE_RE.sub(' ', text)
  text = BAD_SYMBOLS_RE.sub('', text)
  text = MEASUREMENTS.sub('',text)
  text = ' '.join(wnl.lemmatize(word) for word in text.split() if word not in STOPWORDS) # delete stopwors from text
  return text

def preprocess_recipe(name, ingredients, instructions):     
    cleaned_text = '<T> ' + clean_text(name) + ' <I> ' + clean_text(ingredients) + ' <D> ' + clean_text(instructions)
    return cleaned_text