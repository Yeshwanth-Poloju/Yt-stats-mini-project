import pandas as pd
import googleapiclient.discovery
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from string import punctuation
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import os


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#extracting comments using YouTube API v3
def get_classified_comments(video_id):
    API_KEY = "AIzaSyAYd4_4GhoyW8MVUAMI3QQwrHI9E0_Ud38"
    API_service_name = "youtube"
    API_version = "v3"
    youtube = googleapiclient.discovery.build(API_service_name, API_version, developerKey = API_KEY)

    request = youtube.commentThreads().list(part="snippet,replies", videoId=video_id, maxResults=300)
    response = request.execute()

    all_comments = []
    for i in range(len(response['items'])):
        all_comments.append(response['items'][i]['snippet']['topLevelComment']['snippet']['textOriginal'])
    df = pd.DataFrame(all_comments)
    df.rename(columns={0:'Comments'},inplace=True)
    print(df)

    #data pre-processing
    stop_words = stopwords.words('english')
    lzr = WordNetLemmatizer()

    def text_processing(text):
        #convert text into lowercase
        text = text.lower()
        #remove new line characters in text
        text = re.sub(r'\n',' ',text)
        #remove punctuations from text
        text = re.sub('[%s]' % re.escape(punctuation),"",text)
        #remove references and hashtags from text
        text = re.sub("^a-zA-Z0-9$,.","",text)
        #remove multiple spaces from text
        text = re.sub(r'\s+',' ',text,flags=re.I)
        #remove special characters from text
        text = re.sub(r'\W',' ',text)

        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
        text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

        return text

    data = df.copy()
    data.Comments = data.Comments.apply(lambda text: text_processing(text))
    print(data)

    #calculating polarity of comments
    pol = []
    for i in data.Comments.values:
        try:
            analysis = TextBlob(i)
            pol.append(analysis.sentiment.polarity)
        except:
            pol.append(0)

    data['polarity'] = pol

    processed_data = data.copy()
    relevancy = []
    for value in processed_data['polarity']:
        if value > 0.5 or value < -0.5: 
            relevancy.append('Relevant')
        else:
            relevancy.append('Irrelevant')
    processed_data['relevancy'] = relevancy
    #Converting the polarity values from continuous to categorical
    processed_data['polarity'][data.polarity==0] = 0
    processed_data['polarity'][data.polarity > 0] = 1
    processed_data['polarity'][data.polarity < 0] = -1

    sentiment = []
    for value in processed_data['polarity']:
        if value == 1: 
            sentiment.append('positive')
        elif value == 0:
          sentiment.append('neutral')
        else:
            sentiment.append('negative')
    processed_data['sentiment'] = sentiment
    print(processed_data)

  

    #visualization of comments
    processed_data.sentiment.value_counts().plot.bar()
    plt.xlabel("Classification of Comments")
    plt.ylabel("No. of Comments")
    print(data.polarity.value_counts())
    strFile = r"C:\Users\CHANDUPOLOJU\Downloads\PNG_transparency_demonstration_1.png"
    if os.path.isfile(strFile):
      os.remove(strFile)
    plt.savefig(strFile)
    plt.show()

    processed_data.relevancy.value_counts().plot.bar()
    plt.xlabel("Classification of Comments")
    plt.ylabel("No. of Comments")
    print(processed_data.relevancy.value_counts())
    strFile2 = r"C:\Users\CHANDUPOLOJU\Downloads\beautiful-hologram-water-color-frame-png_119551.jpg"
    if os.path.isfile(strFile2):
        os.remove(strFile2)
    plt.savefig(strFile2)
    plt.show()

    #Model construction and training
    corpus = []
    for sentence in data['Comments']:
        corpus.append(sentence)

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(max_features=150)
    X = cv.fit_transform(corpus).toarray()
    y = processed_data['sentiment'].values

    y2 = processed_data['relevancy'].values
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=None)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Positive, Neutral and Negative Classification: ")
    print(cm)
    #tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    #print(tn, fp, fn, tp)
    nb_score = accuracy_score(y_test, y_pred)
    print('accuracy', nb_score)
    print(classification_report(y_test,y_pred))

    X2_train, X2_test, y2_train, y2_test = train_test_split(X,y2,test_size=0.2,random_state=None)
    model = GaussianNB()
    model.fit(X2_train, y2_train)
    pred = model.predict(X2_test)
    print("Relevant and Irrelevant Classification: ")
    print(confusion_matrix(y2_test, pred))
    print(accuracy_score(y2_test, pred))
    print(classification_report(y2_test, pred))
