from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import h5py
import string
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import emoji
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.toktok import ToktokTokenizer
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess


# defining flask
app = Flask(__name__)

# MODEL_PATH = 'best_model1.hdf5'
pos_labels = ['applikasi','ongkir','pembayaran','pengiriman','produk',
          'promo', 'service','shopping experience']
neg_labels = ['akun','app performance','barang','error','iklan','kredit', 'ongkir','order',
                'pembayaran','pencarian']
tokenizer = Tokenizer(filters='')

def prepare_tokenizer_and_weights(X):
    # tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(X)
            
    weights = np.zeros((len(tokenizer.word_index)+1, 300))
    with open("cc.id.300.vec/cc.id.300.vec", encoding="utf8") as f:
        next(f)
        for l in f:
            w = l.split(' ')
            if w[0] in tokenizer.word_index:
                weights[tokenizer.word_index[w[0]]] = np.array([float(x) for x in w[1:301]])
    return tokenizer, weights
    
MAX_LEN_POS = 75
MAX_LEN_NEG = 87


def preprocess_pos():
    stopwords_all = stopwords.words("indonesian") + stopwords.words("english")
    stopwords_baru = ['sih', 'nya', 'iya', 'tah', 'ok', 'oke','bagus', 'eh', 'nya', 'jelek', 'coba', 'kecewa', 'banget', 'kayak', 'semoga', 'buruk', 'gue', 'kali', 'pas', 'mulu',
                  'sebelah', 'langsung', 'suka', 'maaf', 'sih', 'nya', 'di', 'ada', 'tempat', 'untuk', 'yang', 'ini', 'lagi', 'ya',
                  'saja', 'kok', 'deh', 'kalau', 'dan', 'kan', 'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo','kalo', 'amp', 'biar',
                  'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 
                  'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt','&amp', 'yah', 'ni', 'lg', 'tapi',
                  'bisa', 'tah', 'ya', 'sy', 'aku', 'dong', 'ud', 'dr', 'mn', 'km', 'keren', 'puas', 'sip', 'entar', 'jaya', 'mohon', 'lumayan', 'keren', 'god', 'pok', 'love', 'mntap', 'jiwa', 'okee',
                 'seng', 'okey', 'pokoknya', 'tolong', 'benaran', 'mudahan', 'bro', 'goodd', 'bosku', 'bank', 'bank', 'amin', 'gampang', 'malas', 'berat', 'mending', 'goblok', 'tol', 'mantap']
    stopwords_all = stopwords_all + stopwords_baru

    kamus_alay = pd.read_csv('new_kamusalay.csv', encoding="ISO-8859-1", header=None)

    kamus_alay.loc[len(kamus_alay.index)] = ['onkir', 'ongkir'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['blanja', 'belanja'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['uninstal', 'uninstall'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['exsis', 'eksis'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['verivikasi', 'verifikasi'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['recomended', 'recommended'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['bajuz', 'bagus'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['gw', 'gue'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['casback', 'cashback'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['donlwod', 'download'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['donlod', 'download'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['mw', 'mau'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['chasbcak', 'cashback']
    kamus_alay.loc[len(kamus_alay.index)] = ['puazzz', 'puas'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['puaaas', 'puas'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['mmbntu', 'membantu'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['diblibli', 'blibli'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['gratong', 'gratis ongkir'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['cust', 'customer'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['bibli', 'blibli']
    kamus_alay.loc[len(kamus_alay.index)] = ['mncul', 'muncul']
    kamus_alay.loc[len(kamus_alay.index)] = ['enag', 'enak']
    kamus_alay.loc[len(kamus_alay.index)] = ['tlong', 'tolong']
    kamus_alay.loc[len(kamus_alay.index)] = ['goodapp', 'good app']
    kamus_alay.loc[len(kamus_alay.index)] = ['addres', 'address']
    kamus_alay.loc[len(kamus_alay.index)] = ['adlh', 'adalah']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplikasiny', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplikasix', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplikasiyg', 'aplikasi'] 
    kamus_alay.loc[len(kamus_alay.index)] = ['aplikask', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplikaso', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplilasi', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplk', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aplkasi', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['aamsung', 'samsung']
    kamus_alay.loc[len(kamus_alay.index)] = ['apl', 'aplikasi']
    kamus_alay.loc[len(kamus_alay.index)] = ['cant', "can't"]
    kamus_alay.loc[len(kamus_alay.index)] = ['ux', 'user-experience']


    indonesian_lexicon = pd.read_csv('lexicon.csv', encoding="ISO-8859-1", header=None)
    indonesian_lexicon= indonesian_lexicon[[0, 1]]

    frames = [kamus_alay, indonesian_lexicon]
  
    kamus = pd.concat(frames)

    kamus_alay_dict = {}

    for i, row in kamus.iterrows():
        kamus_alay_dict[row[0]] = row[1]

    bins = [0, 2, 5]
    names = ['Negative', 'Positive']
    exclist = string.digits + string.punctuation

    def preprocess(s):
        s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s) 
        s = s.lower()
        s = re.sub(r'&gt|&lt', ' ', s)
        # letter repetition (if more than 2)
        s = re.sub(r'([a-z])\1{2,}', r'\1', s)
        # non-word repetition (if more than 1)
        s = re.sub(r'([\W+])\1{1,}', r'\1', s)
        # phrase repetition
        s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
        s = ' '.join([kamus_alay_dict.get(w, w) for w in s.split()])
        
        return s.strip()

    def remove_stopwords(text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords_all]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords_all]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    def remove_special_characters(oldtext, remove_digits=True):
        table_ = str.maketrans(exclist, ' '*len(exclist))
        newtext = ' '.join(oldtext.translate(table_).split())
        return newtext

    def give_emoji_free_text(text):
        return emoji.get_emoji_regexp().sub(r'', text)

    def clean_data(filtered):
        # filtered = pd.DataFrame(df, columns=['userName', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'sentiment'])
        filtered["reviewCreatedVersion"].fillna("missing", inplace=True)
        filtered.dropna(inplace=True)
        filtered = filtered.reset_index()
        filtered = filtered.drop(columns=['index'])
            
        filtered['preprocessed']=filtered['content'].apply(remove_special_characters)
        filtered['preprocessed']=filtered['preprocessed'].apply(give_emoji_free_text)

        filtered = filtered.dropna()
        filtered = filtered.reset_index()
        filtered = filtered.drop(columns=['index'])
            
        filtered['preprocessed'] = filtered['preprocessed'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2])) 
        filtered['preprocessed'] = filtered['preprocessed'].apply(lambda x: ' '.join([w for w in x.split() if len(w)<12]))
        filtered['preprocessed'] = filtered['preprocessed'].str.lower()
        
        filtered['preprocessed'] = filtered['preprocessed'].apply(remove_stopwords)
        filtered = filtered.replace(r'^\s*$', np.NaN, regex=True)
        filtered = filtered.dropna()
        filtered['preprocessed'] = filtered['preprocessed'].str.lower()
        filtered =  filtered.reset_index()
        filtered = filtered.drop(columns=['index'])
        return filtered

    def tokenize_data(df):
    
        df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df['preprocessed']] 
    
        return df

    # df = clean_data(df)
    # df = tokenize_data(df)

    df2 = pd.read_csv('topic_positive.csv')
    df1 = pd.read_csv('reviews_2.csv')
    df3 = pd.merge(df1, df2, on=['userName', 'at'])
    df = pd.DataFrame(df3, columns=['content', 'reviews', 'score_x', 'reviewCreatedVersion_x', 'at', 'dominant_topic_theme', 'sentiment'])
    df = df.rename(columns={'dominant_topic_theme': 'topic'})
    df = df.rename(columns={'score_x': 'score'})
    df = df.rename(columns={'reviewCreatedVersion_x': 'reviewCreatedVersion'})


    df = clean_data(df)
    df = tokenize_data(df)
    X = df['preprocessed'].values
    tokenizer, weights = prepare_tokenizer_and_weights(X)
    X_seq = tokenizer.texts_to_sequences(X)
    MAX_LEN = max(map(lambda x: len(x), X_seq))
    X_seq = pad_sequences(X_seq, MAX_LEN)
    MAX_ID = len(tokenizer.word_index)
    MAX_LEN_GLOBAL = MAX_LEN

    return df


preprocess_pos()


MODEL_POS = 'tc_pos_2.hdf5'
model_pos = load_model(MODEL_POS)

MODEL_NEG = 'tc_neg_3.hdf5'
model_neg = load_model(MODEL_NEG)

# @app.route('/')

# def home():
#     return render_template("home.html")

@app.route('/', methods =["GET", "POST"])
def positive():
    if request.method == "POST":
        first_name = request.form.get("text_predict")
        sequence = tokenizer.texts_to_sequences([first_name])
        padded = pad_sequences(sequence, MAX_LEN_POS)
        pred = model_pos.predict(padded)
        prediction = pos_labels[np.argmax(pred)]
        return render_template("positive.html", pred=" Topic of the above text is {}".format(prediction))
    return render_template("positive.html")


if __name__ == '__main__':
    app.run(debug=True)