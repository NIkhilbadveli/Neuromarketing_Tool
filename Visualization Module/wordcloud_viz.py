import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def show_wordcloud(comp=True):
    """Generate a wordcloud and show the image"""
    if comp:
        df = pd.read_csv('competitor_keywords_nltk.csv')
    else:
        df = pd.read_csv('influencer_keywords_nltk.csv')

    d = {}
    for n, a, x in df.values:
        d[a] = x
    wordcloud = WordCloud(background_color='white', width=1920, height=1080)
    wordcloud.generate_from_frequencies(frequencies=d)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


show_wordcloud(comp=True)
