"""This module uses Sentence Transformers model to get the sentence embeddings and calculate the similarity score."""
import itertools
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


def split_text_into_sentences(text):
    """Splits a given text into multiple sentences using RegEx"""
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)


def get_average_similarity(model, ps1, ps2):
    """Calculates the average of the similarity score based on all the combinations of these two lists of texts.
    This function uses Sentence Transformers model."""
    sm_scores = []  # list to store similarity scores for each combination of post texts.
    try:
        sembs_1 = model.encode(ps1)  # Sentence embeddings for the first list of posts
        sembs_2 = model.encode(ps2)  # Sentence embeddings for the second list of posts

        # Iterating through each of the first list of embeddings and finding cosine similarity with the second list of embeddings
        for semb in sembs_1:
            sm_scores.extend(cosine_similarity([semb], sembs_2))
    except Exception as e:
        print(e)
        print(ps1)
        print(ps2)
    # Taking the average of the scores. Can take median as well.
    return np.mean(sm_scores)


def get_median_similarity(model, ps1, ps2):
    """Calculates the average of the similarity score based on all the combinations of these two lists of texts.
    Using the Universal sentence encoder model."""
    sm_scores = []
    try:
        sembs_1 = model(ps1).numpy()  # Sentence embeddings for the first list of posts
        sembs_2 = model(ps2).numpy()  # Sentence embeddings for the second list of posts

        for semb in sembs_1:
            sm_scores.extend(cosine_similarity([semb], sembs_2))
    except Exception as e:
        print(e)
        # print(ps1)
        # print(ps2)
    return np.median(sm_scores)


def get_inverse_distance_similarity(model, ps1, ps2):
    """Find the similarity by calculating the overlap between the clusters of both the embeddings.
    Using the Universal sentence encoder model."""
    inverse_distance = 0
    try:
        sembs_1 = model(ps1).numpy()  # Sentence embeddings for the first list of posts
        sembs_2 = model(ps2).numpy()  # Sentence embeddings for the second list of posts

        sembs_1_centroid = np.mean(sembs_1, axis=0)
        # Getting the centroid for the first list of embeddings. Assuming that they form a cluster in the n-dim space.
        sembs_2_centroid = np.mean(sembs_2, axis=0)

        inverse_distance = 1 / distance.cdist([sembs_1_centroid], [sembs_2_centroid])[0][0]
    except Exception as e:
        print(e)
        print(ps1)
        print(ps2)
    return inverse_distance


def get_overlap_similarity(model, ps1, ps2):
    """Find the similarity by calculating the overlap between the clusters of both the embeddings.
    This function uses Sentence Transformers model."""
    similarity_score = 0
    try:
        sembs_1 = model.encode(ps1)  # Sentence embeddings for the first list of posts
        sembs_2 = model.encode(ps2)  # Sentence embeddings for the second list of posts

        # Getting the centroid for the first list of embeddings. Assuming that they form a cluster in the n-dim space.
        sembs_1_centroid = np.mean(sembs_1, axis=0)
        sembs_1_radius = np.max(distance.cdist(sembs_1, [sembs_1_centroid]))
        # Taking the max distance among all the points in the first cluster as the radius of the cluster.

        sembs_2_dists = distance.cdist(sembs_2, [sembs_1_centroid])
        # Getting all the distances of the points in the second cluster from the centroid of the first cluster.
        n_A_or_B = len(sembs_1) + len(sembs_2)  # Union of both the clusters.
        n_A_and_B = len(sembs_2_dists[sembs_2_dists <= sembs_1_radius])
        # Intersection of the clusters is calculated by taking the number of points from the second cluster that are present within the radius of the first cluster.
        similarity_score = round(n_A_and_B / n_A_or_B, 4)  # Ratio of intersection over union.
    except Exception as e:
        print(e)
        print(ps1)
        print(ps2)
    return similarity_score


def get_vito_similarity(model, ps1, ps2, vito_number):
    """Find the similarity by calculating the overlap between the clusters of both the embeddings.
    Using the Universal sentence encoder model."""
    similarity_score = 0
    scaler = StandardScaler()
    try:
        sembs_1 = model(ps1).numpy()  # Sentence embeddings for the first list of posts
        sembs_2 = model(ps2).numpy()  # Sentence embeddings for the second list of posts

        sembs_1 = scaler.fit_transform(sembs_1)
        sembs_2 = scaler.fit_transform(sembs_2)

        sembs_1_centroid = np.mean(sembs_1, axis=0)
        sembs_2_centroid = np.mean(sembs_2, axis=0)

        similarity_score = distance.cdist([sembs_1_centroid], [sembs_2_centroid])[0][0] / vito_number
    except Exception as e:
        print(e)
        print(ps1)
        print(ps2)
    return similarity_score


def get_vito_number(model, text_corpus):
    """Find the maximum possible distance in the n-dim hyperspace."""
    sembs = model(text_corpus).numpy()
    # print(sembs.shape)
    scaler = StandardScaler()
    sembs = scaler.fit_transform(sembs)
    # print(np.max(sembs, axis=0) - np.min(sembs, axis=0))
    return np.sqrt(np.sum((np.max(sembs, axis=0) - np.min(sembs, axis=0)) ** 2))


def competitor_influencer_similarity():
    """Do the similarity analysis between all the competitors and influencers"""
    df_competitors = pd.read_csv('preprocessed_data_comp_without_punc.csv')
    df_influencers = pd.read_csv('preprocessed_data_inf_without_punc.csv')
    # Dropping the null values if there are any
    df_influencers.dropna(inplace=True)
    df_competitors.dropna(inplace=True)
    df_competitors = df_competitors[df_competitors['platform'] != 'website']

    # Loading the sentence transformers model.
    # bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    # vito_number = get_vito_number(use_model, df_competitors['text'].tolist())
    # print('Vito number', vito_number)

    # Getting unique values of competitor and influencer names as a list
    cmptr_list = df_competitors['competitor_name'].unique().tolist()
    inflcr_list = df_influencers['influencer_username'].unique().tolist()

    final_scores = []
    # Iterating through all the competitors
    for cmptr in cmptr_list:
        # Getting all the posts belonging to the competitor given by 'cmptr'
        cmptr_posts = df_competitors.loc[df_competitors['competitor_name'] == cmptr, 'text'].tolist()
        # Iterating through all the influencers
        for inflcr in inflcr_list:
            # Getting all the posts belonging to the influencer given by 'inflcr'
            inflcr_posts = df_influencers.loc[df_influencers['influencer_username'] == inflcr, 'text'].tolist()
            print('Finding similarity between the competitor', cmptr, ' and the influencer', inflcr, '...')
            # Appending to the list of similarity scores for each combination of competitor and influencer.
            final_scores.append(
                (cmptr, inflcr, get_median_similarity(model=use_model, ps1=cmptr_posts, ps2=inflcr_posts)))

            # Saving the results for every iteration.
            df_similarity = pd.DataFrame(final_scores,
                                         columns=['competitor_name', 'influencer_name', 'similarity_score'])
            df_similarity.to_csv('competitor_influencer_similarity.csv', index=False)


def competitor_competitor_similarity(sentence_split=False):
    """Do the similarity analysis between all the competitors"""
    df_competitors = pd.read_csv('preprocessed_data_comp_without_punc.csv')
    df_competitors.dropna(inplace=True)
    df_competitors = df_competitors[df_competitors['platform'] != 'website']

    # Loading the sentence transformers model.
    # bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # vito_number = get_vito_number(use_model, df_competitors['text'].tolist())
    # print('Vito number', vito_number)

    # Getting unique values of competitor and influencer names as a list
    cmptr_list = df_competitors['competitor_name'].unique().tolist()

    final_scores = []
    # Iterating through all the combinations of competitors
    for ns in itertools.combinations(cmptr_list, 2):
        cmptr1, cmptr2 = ns[0], ns[1]
        # Skipping for the same competitor combination
        if cmptr1 == cmptr2:
            continue

        # Getting all the posts belonging to the competitor given by 'cmptr1'
        cmptr1_posts = df_competitors.loc[df_competitors['competitor_name'] == cmptr1, 'text'].tolist()
        cmptr2_posts = df_competitors.loc[df_competitors['competitor_name'] == cmptr2, 'text'].tolist()

        # If sentence split is enabled, will split the posts into sentences.
        if sentence_split:
            stcs = []
            for post in cmptr1_posts:
                stcs.extend(split_text_into_sentences(post))
            cmptr1_posts = stcs.copy()

            stcs = []
            for post in cmptr2_posts:
                stcs.extend(split_text_into_sentences(post))
            cmptr2_posts = stcs.copy()

        print('Finding similarity between the competitor 1', cmptr1, ' and the competitor 2', cmptr2, '...')
        # Appending to the list of similarity scores for each combination of competitors
        final_scores.append(
            (cmptr1, cmptr2,
             get_median_similarity(model=use_model, ps1=cmptr1_posts, ps2=cmptr2_posts)))

        df_similarity = pd.DataFrame(final_scores, columns=['competitor_1', 'competitor_2', 'similarity_score'])
        df_similarity.to_csv('competitor_competitor_similarity.csv', index=False)


# competitor_competitor_similarity(sentence_split=False)
competitor_influencer_similarity()
