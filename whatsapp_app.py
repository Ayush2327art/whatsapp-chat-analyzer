import streamlit as st
import pandas as pd
import re
from collections import Counter
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
nltk.download('punkt')

st.set_page_config(layout="wide")
st.title("📱 WhatsApp Chat Analyzer")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload WhatsApp chat (.txt)", type="txt")

# --- Function to parse WhatsApp chat ---
def parse_chat(chat_lines):
    pattern = r'(\d{2}/\d{2}/\d{4}), (\d{1,2}:\d{2} (?:AM|PM)) - ([^:]+): (.+)'
    data = []
    for line in chat_lines:
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            data.append([date, time, sender.strip(), message.strip()])
    df = pd.DataFrame(data, columns=['Date', 'Time', 'Sender', 'Message'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.day_name()
    df['Month'] = df['Datetime'].dt.month_name()
    return df

# --- Process Uploaded File ---
if uploaded_file:
    chat_text = uploaded_file.read().decode('utf-8').splitlines()
    df = parse_chat(chat_text)

    st.subheader("🧾 Chat Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", len(df))
    col2.metric("Participants", df['Sender'].nunique())
    col3.metric("Media Shared", (df['Message'] == '<Media omitted>').sum())
    col4.metric("Links Shared", df['Message'].str.contains('http').sum())

    st.subheader("🌥 Word Cloud")
    filtered_msgs = df[~df['Message'].str.contains('<Media omitted>|http')]
    text = ' '.join(filtered_msgs['Message'].tolist())
 if text.strip():  # Only proceed if text has content
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.subheader("🌥 Word Cloud of Frequent Words")
    st.pyplot(fig)
else:
    st.warning("⚠️ Not enough words to generate a Word Cloud. Please upload a longer chat file.")

    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("👥 Most Active Users")
    top_senders = df['Sender'].value_counts().head(10)
    st.bar_chart(top_senders)

    st.subheader("📅 Messages Over Time")
    daily_msgs = df.groupby('Date').size()
    st.line_chart(daily_msgs)

    st.subheader("🔥 Activity Heatmap")
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(ordered_days)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("😊 Emoji Usage")
    emojis = []
    for msg in df['Message']:
        emojis += [c for c in msg if c in emoji.EMOJI_DATA]
    emoji_count = dict(Counter(emojis).most_common(5))
    if emoji_count:
        fig, ax = plt.subplots()
        ax.pie(emoji_count.values(), labels=emoji_count.keys(), autopct="%1.1f%%")
        st.pyplot(fig)

    st.subheader("🧠 Most Discussed Topics")
    words = [word.lower() for msg in filtered_msgs['Message'] for word in msg.split() if len(word) > 3 and word.isalpha()]
    common_words = Counter(words).most_common(10)
    st.write(pd.DataFrame(common_words, columns=["Topic", "Count"]))

    st.subheader("😄 Sentiment Analysis")
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    filtered_msgs['Sentiment'] = filtered_msgs['Message'].apply(get_sentiment)
    sentiment_counts = pd.cut(filtered_msgs['Sentiment'], bins=[-1, -0.05, 0.05, 1], labels=["Negative", "Neutral", "Positive"]).value_counts()
    st.bar_chart(sentiment_counts)

else:
    st.info("Please upload a WhatsApp chat `.txt` file to get started.")
