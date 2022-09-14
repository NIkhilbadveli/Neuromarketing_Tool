"""
This file consists of functions we will use to fetch the data from different platforms.
And the data should be saved in the folder data/_file_name_.csv. We should decide on the format of the data.

_file_name_ = platform name + time period + user_name
"""
import json
import threading
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import tweepy
import yaml
from instaloader import Instaloader, Profile
from facebook_scraper import FacebookScraper
import trafilatura
from trafilatura.spider import focused_crawler


class ThreadWithResult(threading.Thread):
    """
    Helper class that can return a result from a thread.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def fetch_from_twitter(user_name, MAX_TWEETS=2000):
    """
    Fetch tweet data from Twitter. We can use the Twitter API to fetch the data.
    Tweepy library:- https://docs.tweepy.org/en/latest/getting_started.html#introduction
    Could use https://github.com/bisguzar/twitter-scraper to scrape the data.
    SO:- https://stackoverflow.com/questions/65694023/is-there-a-way-to-fetch-tweets-from-a-specific-user-during-a-specific-time-frame
    """
    # Get twitter tokens from api_config.json
    with open('api_config.json') as json_file:
        twitter_config = json.load(json_file)['twitter']

    # Initialize the twitter API using tweepy
    client = tweepy.Client(bearer_token=twitter_config['bearer_token'])

    # Get user details
    user = client.get_user(username=user_name)

    # Use paginator to get the last 2000 tweets
    user_tweets_gen = tweepy.Paginator(client.get_users_tweets, id=user.data.id,
                                       tweet_fields=['created_at', 'context_annotations', 'geo', 'entities',
                                                     'public_metrics'], max_results=100).flatten(limit=MAX_TWEETS)

    print('Fetching tweets from {}...'.format(user_name))
    user_tweets = [tweet for tweet in user_tweets_gen]
    print('Fetching tweets from {}...Done'.format(user_name))

    # Save the user_tweets in .csv file after converting to dataframe
    user_tweets_df = pd.DataFrame(user_tweets)
    user_tweets_df.to_csv('data/Twitter/' + user_name + '_twitter_posts' + '.csv', index=False)


def fetch_from_facebook(fb_scraper, user_name, MAX_POSTS=400):
    """
    Fetch facebook data from Facebook. We can use the Facebook API to fetch the data.
    """

    def get_comments(pst_id, cmt_list, MAX_COMMENTS=40):
        """
        This is a helper function to get the comments from a post.
        :param pst_id:
        :param cmt_list:
        :param MAX_COMMENTS:
        :return:
        """
        comments = []
        for cmt in cmt_list:
            comments.append(
                (pst_id, cmt['comment_time'].strftime('%Y-%m-%d %H:%M:%S'), cmt['comment_text'], cmt['commenter_name']))
            if len(comments) >= MAX_COMMENTS:
                break
        return comments

    # Fetch posts using the facebook_scraper library
    scraped_posts = []
    scraped_comments = []
    print('Fetching posts from Facebook for {}...'.format(user_name))
    for post in fb_scraper.get_posts(user_name, pages=None, extra_info=True, options={'comments': False}):
        scraped_posts.append((post['post_id'], post['post_url'], post['time'].strftime('%Y-%m-%d %H:%M:%S'),
                              post['username'], post['text'], post['likes'], post['comments']))
        # scraped_comments.extend(get_comments(post['post_id'], post['comments_full']))

        if len(scraped_posts) >= MAX_POSTS:
            break
    print('Fetching posts from Facebook for {}...Done'.format(user_name))

    # Save the scraped posts & comments in .csv file after converting to dataframe
    df_posts = pd.DataFrame(scraped_posts,
                            columns=['post_id', 'post_url', 'date', 'username', 'text', 'likes', 'comments'])
    df_posts.to_csv('data/Facebook/' + user_name + '_facebook_posts' + '.csv', index=False)

    # df_comments = pd.DataFrame(scraped_comments, columns=['post_id', 'date', 'text', 'commenter_name'])
    # df_comments.to_csv('data/' + user_name + '_facebook_comments' + '.csv')


def fetch_from_instagram(L, user_name, MAX_POSTS=200):
    """
    Fetch instagram data from Instagram. We can use the Instagram API to fetch the data.
    """

    def get_comments(pst, MAX_COMMENTS=200):
        """
        This is a helper function to get the comments from a post.
        :param pst:
        :param MAX_COMMENTS:
        :return:
        """
        comments = []
        for cmt in pst.get_comments():
            comments.append(
                (pst.mediaid, cmt.created_at_utc.strftime('%Y-%m-%d %H:%M:%S'), cmt.text, cmt.owner.username))
            if len(comments) >= MAX_COMMENTS:
                break
        return comments

    # Get the profile of the requested user
    profile = Profile.from_username(L.context, username=user_name)

    # Get the posts of the user
    # Do not use the Instagram app or another instance of InstaLoader() while scraping the data.
    scraped_posts = []
    scraped_comments = []
    print('Fetching posts from Instagram for {}...'.format(user_name))
    for post in profile.get_posts():
        scraped_posts.append((post.mediaid, post.date, post.owner_username, post.caption, post.likes, post.comments))
        # scraped_comments.extend(get_comments(post))

        if len(scraped_posts) >= MAX_POSTS:
            break
    print('Fetching posts from Instagram for {}...Done'.format(user_name))

    # Save the scraped data in two .csv files - one for posts and the other for comments
    df_posts = pd.DataFrame(scraped_posts,
                            columns=['mediaid', 'date', 'owner_username', 'caption', 'likes', 'comments'])
    df_posts.to_csv('data/Instagram' + user_name + '_instagram_posts' + '.csv', index=False)

    # df_comments = pd.DataFrame(scraped_comments, columns=['mediaid', 'date', 'text', 'commenter_username'])
    # df_comments.to_csv('data/' + user_name + '_instagram_comments' + '.csv')


def fetch_from_youtube(user_name):
    """
    Fetch YouTube data from YouTube. We can use the YouTube API to fetch the data.
    """
    pass


def fetch_from_tiktok(user_name):
    """
    Fetch TikTok data from TikTok. We can use the TikTok API to fetch the data.
    """
    pass


def fetch_from_website(competitor_name, url):
    """
    Fetch data from a website. We have to use a scraping library to fetch the data. (BeautifulSoup, Scrapy, BrightData, etc.)
    """

    def extract_text_from_single_web_page(single_page_url):
        """
        This is a helper function to extract the text from a single web page.
        :param single_page_url:
        :return:
        """
        downloaded_url = trafilatura.fetch_url(single_page_url)
        a = trafilatura.extract(downloaded_url,
                                trafilatura.extract(downloaded_url, output=True, include_comments=False))
        return a

    print('Fetching data from the website {}...'.format(url))
    # starting a crawl
    to_visit, known_urls = focused_crawler(url)

    # Convert to_visit to list and sort the known_urls
    to_visit, known_urls = list(to_visit), sorted(known_urls)

    data = []
    exception_urls = []
    for page_url in to_visit:
        try:
            data.append(extract_text_from_single_web_page(page_url))
        except Exception as e:  # Todo: Convert to specific exception
            exception_urls.append(page_url)
            # print(e)
    print('Fetching data from the website {}...Done'.format(url))

    # Save the scraped data in a .csv file
    # df = pd.DataFrame(data, columns=['url', 'text'])
    # df.to_csv('data/' + competitor_name + '_website_text' + '.csv')

    # Save the scraped data in a .txt file
    with open('data/Website' + competitor_name + '_website_text' + '.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(data))


# Write the main function here
def main():
    # Get the list of competitors & influencers from run_config
    competitors = pd.read_csv('competitors.csv')
    influencers = pd.read_csv('influencers.csv')

    # Get insta config from api_config.json
    with open('api_config.json') as json_file:
        insta_config = json.load(json_file)['instagram']
        fb_config = json.load(json_file)['facebook']

    # Initialize the instagram API using instaloader
    L = Instaloader()

    # Login to instagram
    L.login(user=insta_config['username'], passwd=insta_config['password'])

    # Initialize the facebook API using facebook_scraper
    fb_scraper = FacebookScraper()

    # Login to facebook
    fb_scraper.login(fb_config['email'], fb_config['password'])

    # Create a dict of options
    options = {
        'twitter': fetch_from_twitter,
        'fb': fetch_from_facebook,
        'insta': fetch_from_instagram,
        'youtube': fetch_from_youtube,
        'tiktok': fetch_from_tiktok,
        'website': fetch_from_website
    }

    # Facebook: login required - puresportclubs, formnutrition
    for key, value in options:
        # Here, 'value' is the function to be called
        if key == 'fb':
            print('Fetching data from Facebook...')
            for idx, competitor in competitors.iterrows():
                value(fb_scraper, competitor[key])
            print('Fetching data from Facebook...Done')
        elif key == 'insta':
            print('Fetching data from Instagram...')
            for idx, competitor in competitors.iterrows():
                value(L, competitor[key])
            print('Fetching data from influencers...')
            for idx, influencer in influencers.iterrows():
                value(L, influencer[key])
            print('Fetching data from influencers...Done')
            print('Fetching data from Instagram...Done')
        elif key == 'twitter':
            print('Fetching data from Twitter...')
            for idx, competitor in competitors.iterrows():
                value(competitor[key])
            print('Fetching data from Twitter...Done')
        elif key == 'website':
            print('Fetching data from Website...')
            for competitor in competitors.iterrows():
                value(competitor['name'], competitor[key])
            print('Fetching data from Website...Done')
        else:
            print('Invalid platform')


if __name__ == '__main__':
    s = 0
    data_thread = ThreadWithResult(target=main, daemon=True)
    data_thread.start()
    while data_thread.is_alive():
        print('Fetching the data...', str(timedelta(seconds=s)))
        s += 30
        time.sleep(30)
