from flask import Flask, render_template
import json
import plotly
from visualize import Visualizer
import threading

app = Flask(__name__)

@app.route("/")
def main():
    viz = Visualizer()
    tweet_numbers = viz.number_tweets()
    tweet_lengths = viz.len_tweets()
    top_trigrams = viz.top_trigrams()
    assert viz.word_cloud()
    five_topics = viz.five_topics()
    positivity = viz.get_positivity()
    negativity = viz.get_negativity()
    activity = viz.get_activity()
    max_tweets = viz.get_maxtweets()
    sentiment_map = viz.sentiment_topic_map()
    ticker = viz.get_ticker()
    tweetnum_graph = json.dumps(tweet_numbers, cls=plotly.utils.PlotlyJSONEncoder)
    tweetlen_graph = json.dumps(tweet_lengths, cls=plotly.utils.PlotlyJSONEncoder)
    trigram_graph = json.dumps(top_trigrams, cls=plotly.utils.PlotlyJSONEncoder)
    top_topics = json.dumps(five_topics, cls=plotly.utils.PlotlyJSONEncoder)
    negativity = json.dumps(negativity, cls=plotly.utils.PlotlyJSONEncoder)
    positivity = json.dumps(positivity, cls=plotly.utils.PlotlyJSONEncoder)
    activity = json.dumps(activity, cls=plotly.utils.PlotlyJSONEncoder)
    max_tweets = json.dumps(max_tweets, cls=plotly.utils.PlotlyJSONEncoder)
    s_t_map = json.dumps(sentiment_map, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', tweetnum=tweetnum_graph, tweetlen=tweetlen_graph,
                           trigram=trigram_graph, top_topics=top_topics, negativity=negativity,
                           positivity=positivity, activity=activity, max_tweets=max_tweets, s_t_map=s_t_map,
                           ticker=ticker)



if __name__ == "__main__":
    app.run()
