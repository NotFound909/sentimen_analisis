import streamlit as st
import pandas as pd
import re
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
from textblob import TextBlob
import numpy as np
import pickle

# Load Naive Bayes model
filename = 'naive_bayes_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Preprocessing functions
nor = {'bung':'','gak':'tidak','ghaza':'','nt':'nicetry','tdk':'tidak','papa':'apa-apa','adek':'adik','thailand':'','nguyen':'','palestina':'','israel':'','udh':'sudah','coack':'coach',
       'diskon':'','kucing':'','%':'','rp':'','cute':'','!':'','isr4el':'','streaming':'','live':'','link':'','euro':'','btw':'','prancis':'','belgia':'','mau':'ingin','ttp':'tetap',
       'ngepoor':'','brok':'','bgt':'banget','jancok':'','anjgg':'','gabsa':'tidak bisa','aus':'australia','jgn':'jangan','mesi':'','org':'orang','australiatralia':'','timnasday':''}

def normalisasi(text):
  for key, value in nor.items():
    text = text.replace(key, value)
  return text

# Load key normalization data
key_norm = pd.read_csv('key_norm.csv')
def text_normalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
  text = str.lower(text)
  return text

# Stopword removal
more_stop_words = ['semangat','sudah','wasit','pelatih','yg', 'scroll', 'iih', 'yaaahh', 'lahhh', 'niee', 'wahh', 'hihiw', 'coii', 'dongg', 'aih', 'huss', 'yha',
'hehew', 'cie', 'wuaaaaaaa', 'nahh', 'kkkk', 'ohhh', 'hwahahahahah', 'yh', 'vaaayy', 'ehehehhe', 'cengo',
'kahh', 'mwhehehe', 'xixix', 'hdjshsjdj', 'hihi', 'huhuhu', 'atulah', 'imo', 'looooooo', 'abyss', 'wkwkwjw',
'ahskwhskja', 'woe', 'wkwkeek', 'hmmmm', 'cuna', 'skz', 'eak', 'c', 'ehhhh', 'wkkw', 'ugh', 'donk', 'mipii',
'heum', 'wkwkk', 'iai', 'prass', 'siak', 'kons', 'hhhhhh', 'mhhh', 'ksbsksbsk', 'nyaa', 's', 'amp', 'shsjaksjs',
'wkwkw', 't', 'emm', 'gc', 'yahhh', 'ceunah', 'etdah', 'yapp', 'xiaohong', 'hahahahahahaahahahahaha', 'hahahaha',
'aaaaa', 'x', 'noh', 'eben', 'uwuwuwuwu', 'cm', 'perhhh', 'hahahaahaha', 'hiss', 'pokokmeeeennn', 'tuuu', 'blabla',
'slh', 'wey', 'jai', 'huweeeeee', 'e', 'hiksrot', 'askjfhagh', 'asp', 'duper', 'seh', 'x', 'xi', 'dahlah', 'xii',
'woilah', 'noh', 'lahaula', 'yup', 'cijejes', 'siot', 'muk', 'sak', 'dol', 'ups', 'snsnnd', 'iyhh', 'kkk', 'haiiiii',
'wew', 'iaaa', 'aaaa', 'haaa', 'toh', 'rinn', 'hiks', 'euy', 'sksjsk', 'deh', 'yakmat', 'nyah', 'hm', 'ann', 'yailah',
'yeaa', 'ae', 'ko', 'lah', 'an', 'doang', 'ahah', 'ii', 'ahahahaha', 'ikonics', 'ekke', 'boom', 'sg', 'hish',
'vkfjdkkd', 'horeeeeeeee', 'yaaah', 'hah', 'namjun', 'of', 'syi', 'hmm', 'revmen', 'dips', 'oioi', 'hft', 'heeseung',
'eng', 'heh', 'ava', 'au', 'dong', 'yaaa', 'sihhh', 'reposted', 'kah', 'kok', 'se', 'lho', 'deh', 'yng', 'weh', 'the',
'mah', 'woiii', 'way', 'm', 'leee', 'kasi', 'gue', 'say', 'ges', 'lagi', 'b', 'r', 'ssssst', 'yaa', 'hots', 'to',
'haaahaaa', 'asp', 'brb', 'dan', 'ya', 'wah', 'oe', 'si', 'huuffhh', 'ye', 'lanyalla', 'dg', 'rt', 'dgn', 'noh', 'ama',
'duh', 'deh', 'brb', 'ny', 'nct', 'svt', 'hm', 'hmmm', 'like', 'you', 'don\'t', 'juga', 'mcm', 'reeeeekk',
'bangeeeeeeeeddd', 'd', 'haaahaaa', 'weey', 'w', 'belamra', 'susar', 'nich', 'yujiem', 'need', 'socialize', 'kan',
'deng', 'tos', 'nah', 'mau', 'mder', 'ad', 'wkwkwkwkwk', 'dat', 'ayolohhh', 'okeh', 'co', 't', 'ngam', 'microsleep',
'hahahah', 'xpa', 'kau', 'moment', 'ampe', 'donk', 'ehehehhe', 'spt', 'fuhh', 'wkwkwkwk', 'wkwkwkwkwkwkwkwk',
'morningg', 'dari', 'uy', 'sem', 'ahh', 'amp', 'biar', 'bikin', 'bilang', 'sm', 'je', 'guys', 'krn', 'nya', 'nih',
'sih', 'sch', 'plis', 'kulon', 'hufftttt', 'aaaahhhh', 'bangsaadd', 'yuk', 'swipe', 'gw', 'kr', 'bngt', 'dr', 'ohh',
'lhoo', 'bist', 'das', 'wkwk', 'ya', 'doang', 'der', 'nge', 'dich', 'weey', 'nk', 'h', 'v', 'AI', 'woi', 'mek', 'be',
'like', 'yailah', 'yang', 'hee', 'cin', 'wkwkwk', 'k', 'bwt', 'hehe', 'buat', 'sqw', 'i', 'hahahaahaha', 'feel', 'u',
'nder', 'ini', 'haha', 'hihi', 'halloo', 'donggg', 'di', 'sksks', 'esok', 'banget', 'karena', 'hwhwhw', 'wkwkww',
'was', 'hehehe', 'eh', 'hahaha', 'how', 'miss', 'clinical', 'years', 'c', 'dah', 'oi', 'nak', 'miin', 'dh', 'ii', 'gi',
'heyy', 'hey', 'tpp', 'sii', 'tpp', 'lg', 'lagi', 'wkt', 'ngan', 'waktu', 'yarrr', 'jreng', 'terus', 'she', 'was',
'pls', 'w', 'uda', 'pun', 'la', 'wkwk', 'yeayyyyy', 'wkwkwk', 'ni', 'i', 'k', 'cm', 'lt', 'why', 'heu', 'ke', 'tuh',
'aja', 'saja', 'ak', 'kek', 'abis', 'gt', 'gitu', 'wkwkwc', 'tp', 'xixixixi', 'mmg', 'memang', 'pi', 'dah', 'yah',
'lt', 'deh', 'yth', 'pd', 'tu', 'tuuu', 'pada', 'emg', 'bln', 'bsk', 'bnr', 'kl', 'kt', 'mmf', 'diaa', 'kalii',
'jbjb', 'gtu', 'yak', 'kyk', 'plk', 'kyg', 'elah', 'thn', 'ah', 'syahla', 'xixixixi', 'brooooooo', 'bruuuhhhh',
'cmon', 'alias', 'dll', 'plk', 'eehh', 'hyung', 'pas', 'oh', 'toh', 'sksjsk', 'hiss', 'bang', 'bs', 'org', 'huhu',
'wkwkwkkw', 'wkekw', 'wkekwkkk', 'braw', 'sing', 'yakmat']  # Add your list of stopwords
stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)
new_array = ArrayDictionary(stop_words)
stop = StopWordRemover(new_array)
def stopwords(str_text):
  str_text = stop.remove(str_text)
  return str_text

# Tokenization
def tokenize(text):
  return text.split()

# Stemming
def stem(text_cleaning):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  kd = []
  for i  in text_cleaning:
    d = stemmer.stem(i)
    kd.append(d)
  kata_clean = []
  kata_clean = ' '.join(kd)
  return kata_clean

# Translation
def convert_to_english(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='id', dest='en')
        return translated.text
    except Exception as e:
        print(f"Error translating text: {text} - {e}")
        return text

# Sentiment prediction
def predict_sentiment(text):
  # Preprocess the text
  text = normalisasi(text)
  text = text_normalize(text)
  text = stopwords(text)
  text = tokenize(text)
  text = stem(text)
  text = convert_to_english(text)

  # Extract polarity using TextBlob
  polarity = TextBlob(text).sentiment.polarity
  polarity_array = np.array(polarity).reshape(1, -1)

  # Predict sentiment using loaded model
  prediction = loaded_model.predict(polarity_array)[0]
  return prediction

# Streamlit app
st.title("Sentiment Analysis App")

text_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze"):
  prediction = predict_sentiment(text_input)
  st.write(f"Sentiment: {prediction}")
