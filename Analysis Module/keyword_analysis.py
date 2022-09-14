import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from rake_nltk import Rake
import nltk
from nltk.probability import FreqDist


def split_text_into_sentences(text):
    """Splits a given text into multiple sentences using RegEx"""
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)


def keywords_with_frequency(text_corpus, top_n=20):
    """Calculates and returns the top n keywords according their frequency"""
    vectorizer = CountVectorizer()

    # print(vectorizer.get_feature_names_out())
    X = vectorizer.fit_transform(text_corpus)
    idxs = np.argsort(X.toarray(), axis=1)[:, -top_n:]  # Getting the indices of top_n values
    out = []
    for idx in idxs.reshape(-1, 1):
        out.append((vectorizer.get_feature_names_out()[idx][0],
                    X.toarray()[0, idx][0]))  # Getting the word and count for each of the top_n indices.
    df_out = pd.DataFrame(reversed(out), columns=['Word', 'Count'])

    df_out.to_csv('keywords_with_frequency.csv', index=False)


def keywords_with_tfidf(text_corpus, top_n=20):
    """Calculates and returns the top n keywords according their Tf-Idf score"""
    tf_idf_vectorizer = TfidfVectorizer()

    # print(vectorizer.get_feature_names_out())
    X = tf_idf_vectorizer.fit_transform(text_corpus)
    idxs = np.argsort(X.toarray(), axis=1)[:, -top_n:]  # Getting the indices of top_n values
    out = []
    for idx in idxs.reshape(-1, 1):
        out.append((tf_idf_vectorizer.get_feature_names_out()[idx][0],
                    X.toarray()[0, idx][0]))  # Getting the word and count for each of the top_n indices.
    df_out = pd.DataFrame(reversed(out), columns=['Word', 'Count'])

    df_out.to_csv('keywords_with_tfidf.csv', index=False)


def keywords_with_rake(cmptr_name, text_corpus, sentence_split=False, top_n=20):
    """Calculates and returns the top n keywords using Rapid Automatic Keyword Extraction algorithm."""
    r = Rake()  # Initialize the RAKE
    if sentence_split:
        r.extract_keywords_from_sentences(split_text_into_sentences(text_corpus))
    else:
        r.extract_keywords_from_text(text_corpus)

    # Getting top_n keywords based on the rake score.
    return [(cmptr_name,) + elem for elem in r.get_ranked_phrases_with_scores()[:top_n]]


def keywords_with_nltk(cmptr_name, text_corpus, top_n=20):
    """Calculates and returns the top n keywords using NLTK. Assumes that text_corpus is in lowercase
    :param cmptr_name: Name of the competitor
    :param top_n: Top n keywords to be returned
    :param text_corpus: The text to be processed in the form of a string.
    """
    # Make the text lowercase
    text_corpus = text_corpus.lower()

    # Remove numbers and special characters
    text_corpus = re.sub(r'[^a-zA-Z]', ' ', text_corpus)

    # Download the stopwords if not downloaded already
    nltk.download('stopwords', quiet=True)

    # Make a list of english stopwords
    stopwords = nltk.corpus.stopwords.words("english")

    # Extend the stopwords with the custom stopwords
    extra_stopwords = ['https', "get", "the", "day", "one", "our", "amp", "like", "use", 'and', 'your', 'you', 'for',
                       'with', 'this', 'that', 'are', 'can', 'from', 'have', 'all', 'us', 'data',
                       'more', 'what', 'what', 'out', 'how', 'not', 'here', 'but', 'just']
    stopwords.extend(extra_stopwords)

    # Tokenize the text
    tokens = nltk.word_tokenize(text_corpus)

    # Remove the stopwords
    tokens = [token for token in tokens if (token not in stopwords) and (len(token) > 1)]

    # Calculate the frequency of the tokens
    fdist = FreqDist(tokens)

    # Get the top n keywords
    return [(cmptr_name, word, freq) for word, freq in fdist.most_common(top_n)]


def main():
    """Main function that runs through all the competitors data and calculates the top keywords"""
    df_competitors = pd.read_csv('preprocessed_data_comp_without_punc.csv')
    df_influencers = pd.read_csv('preprocessed_data_inf_without_punc.csv')
    # Dropping the null values if there are any
    df_influencers.dropna(inplace=True)
    df_competitors.dropna(inplace=True)
    df_competitors['text'] = df_competitors['text'].astype(str)
    df_influencers['text'] = df_influencers['text'].astype(str)

    cmptr_list = df_competitors['competitor_name'].unique().tolist()
    inflcr_list = df_influencers['influencer_username'].unique().tolist()

    final_output = []
    for cmptr in cmptr_list:
        cmptr_posts = df_competitors.loc[df_competitors['competitor_name'] == cmptr, 'text'].tolist()
        print('Getting keywords for the competitor: {}'.format(cmptr))
        # Call the other keyword functions if needed.
        rps = keywords_with_nltk(cmptr_name=cmptr, text_corpus=' '.join(cmptr_posts), top_n=20)
        final_output.extend(rps)
    df_out = pd.DataFrame(final_output, columns=['Competitor Name', 'Word', 'Count'])
    df_out.to_csv('competitor_keywords_nltk.csv', index=False)

    final_output = []
    for inflcr in inflcr_list:
        inflcr_posts = df_influencers.loc[df_influencers['influencer_username'] == inflcr, 'text'].tolist()
        print('Getting keywords for the influencer: {}'.format(inflcr))
        # Call the other keyword functions if needed.
        rps = keywords_with_nltk(cmptr_name=inflcr, text_corpus=' '.join(inflcr_posts), top_n=20)
        final_output.extend(rps)
    df_out = pd.DataFrame(final_output, columns=['Influencer Name', 'Word', 'Count'])
    df_out.to_csv('influencer_keywords_nltk.csv', index=False)


main()
