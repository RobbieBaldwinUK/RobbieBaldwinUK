import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentAnalysis:
    def __init__(self):
        nltk.download('vader_lexicon')
        self.sid = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        sentiment_scores = self.sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment

# Example usage:
analyzer = SentimentAnalysis()

text = "I absolutely loved the movie! The acting was brilliant and the story was captivating."
sentiment = analyzer.analyze_sentiment(text)
print("Sentiment:", sentiment)

text = "I was disappointed with the service. The staff was rude and unhelpful."
sentiment = analyzer.analyze_sentiment(text)
print("Sentiment:", sentiment)
