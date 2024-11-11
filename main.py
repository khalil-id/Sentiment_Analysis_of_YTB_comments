import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re

# Fonction pour nettoyer le texte
def clean_text(text):
    if isinstance(text, str):
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words("english")])
    else:
        return ''

# Fonction pour filtrer et sauvegarder les données nettoyées
def filter_and_save_clean_data():
    with open("input.csv", encoding="utf-8") as csvfile:  
        reader = csv.reader(csvfile)
        with open("clean.csv", "w", encoding="utf-8") as csv_file:  
            writer = csv.writer(csv_file, delimiter=',')
            for row in reader:
                for string in row:
                    string = re.sub(r'http\S+', '', string)  # Remove URLs
                    string = re.sub(r'\S+@\S+', '', string)  # Remove email addresses
                    emoji_pattern = re.compile("["
                                               u"\U0001F600-\U0001F64F"  # emoticons
                                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                               u"\U00002500-\U00002BEF"  # chinese char
                                               u"\U00002702-\U000027B0"
                                               u"\U00002702-\U000027B0"
                                               u"\U000024C2-\U0001F251"
                                               u"\U0001f926-\U0001f937"
                                               u"\U00010000-\U0010ffff"
                                               u"\u2640-\u2642"
                                               u"\u2600-\u2B55"
                                               u"\u200d"
                                               u"\u23cf"
                                               u"\u23e9"
                                               u"\u231a"
                                               u"\ufe0f"  # dingbats
                                               u"\u3030"
                                               "]+", flags=re.UNICODE)
                    string = emoji_pattern.sub(r'', string)
                    string = string.strip()
                    string = re.sub(r" \d+", " ", string)  # Remove numbers
                    if string.strip() != '':
                        csv_file.write(clean_text(string) + "\n")

def sentiment_analysis():
    comments = pd.read_csv("clean.csv", sep='\t', names=['comment'])
    df = comments['comment'].apply(clean_text)
    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv("clean.csv", index=False)

    data_train = pd.read_csv("DataSet.csv", encoding="latin-1")
    data_testing = pd.read_csv("clean.csv", encoding="utf-8", names=["Comment"])

    labels = data_train.Sentiment
    data_train = data_train.drop(columns=['ItemID'])

    X = data_train.SentimentText
    y = data_train.Sentiment

    X_train, _, y_train, _ = train_test_split(X, y, test_size=.66, random_state=61)
    X_test = data_testing.Comment
    X_test.fillna("", inplace=True)

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(X_train)

    X_train_dtm = vectorizer.transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)

    y_predict_class = nb.predict(X_test_dtm)

    count1 = sum(1 for i in y_predict_class if i == 1)
    count0 = sum(1 for i in y_predict_class if i == 0)

    total = count1 + count0
    if total == 0:
        st.info("No comments for analysis.")
        return

    perc1 = (count1 / total) * 100
    perc0 = (count0 / total) * 100

    st.write("Percentage of positive comments:", perc1)
    st.write("Percentage of negative comments:", perc0)
    st.write("Number of positive comments:", count1)
    st.write("Number of Negative comments:", count0)

    labels = ['Positive', 'Negative']
    sizes = [perc1, perc0]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)  # Pass the figure object directly

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, sizes)
    st.pyplot(fig2) 

    # Create a corrplot
    result_df = pd.DataFrame({
        'positive': [1 if i == 1 else 0 for i in y_predict_class],
        'negative': [1 if i == 0 else 0 for i in y_predict_class]
    })
    fig, ax = plt.subplots()

    sns.heatmap(result_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def extract_video_id(url):
    pattern = r"(?<=v=)[a-zA-Z0-9_-]+"
    match = re.search(pattern, url)
    if match:
        return match.group(0)
    else:
        return None

def get_comments(client, video_id, token=None):
    try:
        response = (
            client.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=token,
            )
            .execute()
        )
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def fetch_comments(api_key, youtube_url):
    vid_id = extract_video_id(youtube_url)

    if not vid_id:
        st.error("Invalid YouTube URL. Please provide a valid URL.")
        return

    output_file = "input.csv"
    yt_client = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page_token = None

    while True:
        resp = get_comments(yt_client, vid_id, next_page_token)

        if not resp:
            break

        comments += resp.get("items", [])
        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    if comments:
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                for comment in comments:
                    text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    csv_writer.writerow([text])
            st.success("Comments fetched and saved to comments.csv")
            
            # Clean data and perform sentiment analysis
            filter_and_save_clean_data()
            sentiment_analysis()
            
        except Exception as e:
            st.error(f"Error writing to CSV file: {e}")
    else:
        st.warning("No comments found for the given video.")

# Streamlit interface
st.title("YouTube Comments Sentiment Analysis")

api_key = "AIzaSyDm5X3ta6c0bTxJPl0C1LbhhY6z2wskY5Q"     #api key: "AIzaSyATH4gtD3kD2y2jFy-jvAKukviyB79vlkU" 
youtube_url = st.text_input("Enter YouTube URL:")

if st.button("Fetch Comments & Analyze Sentiments"):
    if api_key and youtube_url:
        fetch_comments(api_key, youtube_url)
    else:
        st.error("Please provide both YouTube API key and URL.")
