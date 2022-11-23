#!/usr/bin/python

import numpy as np
import pandas as pd
import mysql.connector
import credentials
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


def get_length(line):
    return len(line.split(" "))


class Visualizer:
    def __init__(self):
        # Credentials for database connection
        self.max_tweets = 20000
        self.hostname = credentials.hostname
        self.dbname = credentials.dbname
        self.uname = credentials.uname
        self.pwd = credentials.pwd
        self.stoplist = stopwords.words('english') + credentials.stopwords

    # gets the tweets from DB
    def __get_tweets(self):
        try:
            tweets = []
            mydb = mysql.connector.connect(user=self.uname, password=self.pwd,
                                           host=self.hostname, database=self.dbname,
                                           auth_plugin='mysql_native_password')
            cursor = mydb.cursor()
            cursor.execute("select tweet, sentiment from tweets ORDER BY timestamp DESC")
            result = cursor.fetchall()
            tweets.append([i for i in result])
            cursor.close()
            mydb.close()

            return tweets[0]

        except Exception as e:
            print(e)

    def get_timed_tweets(self):
        # returns tweets with timestamps
        tweets = []
        mydb = mysql.connector.connect(user=self.uname, password=self.pwd,
                                       host=self.hostname, database=self.dbname,
                                       auth_plugin='mysql_native_password')
        cursor = mydb.cursor()
        cursor.execute("select * from tweets  ORDER BY timestamp DESC")
        tweets.append([i for i in cursor])
        cursor.close()
        mydb.close()
        return tweets[0]

    # make pandas dataframe with tweets
    def get_df(self):
        lines = self.get_timed_tweets()
        if len(lines) > 0:
            tweets = pd.DataFrame(lines)
            tweets.columns = [
                'id',
                'userid',
                'raw',
                'tweet',
                'sentiment',
                'topic_0',
                'topic_1',
                'topic_2',
                'topic_3',
                'topic_4',
                'timestamp'
            ]
            tweets['length'] = tweets['tweet'].apply(lambda line: get_length(line))

            return tweets
        else:
            raise Exception(lines)

    # sentiment-topic mapping
    def sentiment_topic_map(self):
        df = self.get_df()
        # nested dictionary of shape: topic_no > sentiment > score
        summary = {}
        positives = ['positive']
        negatives = ['negative']
        for i in range(1, 6):
            summary["topic_{}".format(i)] = {
                    "positive": df.loc[df["sentiment"].isin(positives)]['topic_{}'.format(i - 1)].sum(),
                    "negative": df.loc[df["sentiment"].isin(negatives)]['topic_{}'.format(i - 1)].sum()
            }
        titles = ['Topic_1', 'Topic_2']
        fig = go.Figure(data=[
            go.Bar(
                name='Positive',
                x=titles,
                y=[
                    summary['topic_1']['positive'],
                    summary['topic_2']['positive']
                ],
                marker_color='#75c46f'),
            go.Bar(
                name='Negative',
                x=titles,
                y=[
                    summary['topic_1']['negative'],
                    summary['topic_2']['negative']
                ],
                marker_color='#cc5751')
        ])
        fig.update_layout(barmode='stack',
                          yaxis_title="Number of tweets",
                          margin=dict(l=10, r=10, t=20, b=20)
                          )
        return fig

    # tweet counts over time
    def number_tweets(self):
        tweets = self.get_df()
        print(tweets.shape)
        df = pd.DataFrame(tweets['timestamp'])
        df['number_of_tweets'] = 1
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.set_index(['timestamp'])
        df = df.groupby(pd.Grouper(freq='10Min')).aggregate(np.sum)
        df = pd.DataFrame(df)
        df.reset_index(inplace=True)
        fig = px.line(df, x='timestamp', y='number_of_tweets', template='plotly_dark',
                      title='Number of tweets during the last 24 hours')
        fig.update_xaxes(categoryorder='category descending', title='Date and time (GMT)')\
            .update_yaxes(title='Number of tweets')

        return fig

    # tweets by number of words
    def len_tweets(self):
        fig = px.histogram(self.get_df(), x='length', template='plotly_dark',
                           title='Length (words) of tweets during the last 24 hours',
                           nbins=50)
        fig.update_xaxes(categoryorder='total descending', title='Length')\
            .update_yaxes(title='Number of tweets')
        return fig

    # returns a plot with top 20 trigrams
    def top_trigrams(self):
        stoplist = stopwords.words('english') + ['elon', 'musk', 'elonmusk', 'mamamoo', 'rm', '\n']
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3, 3))
        # matrix of ngrams
        ngrams = c_vec.fit_transform(self.get_df()['tweet'])
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                                ).rename(columns={0: 'frequency', 1: 'trigram'})
        fig = px.bar(df_ngram[:20], x='trigram', y='frequency', title='Top 20 trigrams', template='plotly_dark',
                     color='frequency')
        return fig

    # creates word cloud svg and saves it into file
    def word_cloud(self):
        tweets = self.get_df()
        # create the word cloud and save it
        long_string = " ".join(tweets.tweet.to_list())
        wordcloud = WordCloud(stopwords=self.stoplist,
                              background_color="black",
                              max_words=5000,
                              random_state=42,
                              max_font_size=500)
        wordcloud.generate(long_string)
        wordcloud_svg = wordcloud.to_svg(embed_font=True)
        f = open("static/plots/wordcloud.svg", "w+")
        f.write(wordcloud_svg)
        f.close()

        return True

    # extracts topics from the model and composes a dictionary
    def __get_top_words(self, model, feature_names, n_top_words):
        table = {}
        for topic_idx, topic in enumerate(model.components_):
            table[topic_idx + 1] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

        return table

    # returns the dictionary of top topics
    def five_topics(self):
        tfidf_vectorizer = TfidfVectorizer(stop_words=self.stoplist, ngram_range=(3, 3))
        lda = LatentDirichletAllocation(n_components=2)
        pipe = make_pipeline(tfidf_vectorizer, lda)
        pipe.fit(self.get_df()['tweet'])
        # get a dictionary with top topics
        table = self.__get_top_words(lda, tfidf_vectorizer.get_feature_names(), n_top_words=5)
        # construct a table with top topics
        fig = go.Figure(data=[go.Table(header=dict(values=[i for i in table.keys()]),
                                       cells=dict(values=[i for i in table.values()]))
                              ], layout={"paper_bgcolor": "black"})
        fig.update_layout(
            margin=dict(l=10, r=20, t=10, b=10),
        )
        return fig

    # returns the percentage of the negative tweets
    def get_negativity(self):
        tweets = self.get_df()
        kwords = ['negative']
        neg_tweets = tweets.loc[tweets["sentiment"].isin(kwords)]
        score = (neg_tweets.shape[0] / tweets.shape[0]) * 100

        return score

    # returns the percentage of the negative tweets
    def get_positivity(self):
        tweets = self.get_df()
        kwords = ['positive']
        pos_tweets = tweets.loc[tweets["sentiment"].isin(kwords)]
        score = (pos_tweets.shape[0] / tweets.shape[0]) * 100

        return score

    # returns the number of tweets during the last day
    def get_activity(self):
        tweets = self.get_df()

        return tweets.shape[0]

    # returns the maximum number of tweets for a gauge
    def get_maxtweets(self):
        return self.max_tweets

    # ticker with the labelled tweets
    def get_ticker(self):
        df = self.get_df()
        tweets = []
        for i in range(len(df)):
            fields = {
                "tweet": df.loc[i, "tweet"],
                "sentiment": df.loc[i, "sentiment"],
                "topic_0": str(round(df.loc[i, "topic_0"], 2)),
                "topic_1": str(round(df.loc[i, "topic_1"], 2)),
                "timestamp": df.loc[i, "timestamp"]
            }
            tweets.append(fields)

        return tweets

if __name__ == "__main__":
    test = Visualizer()
    t = test.number_tweets()

