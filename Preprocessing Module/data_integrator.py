import pandas as pd
import numpy as np
import glob
import os
import re


def combine_fb_data(filename):
    """Read fb data into a dataframe and rename the columns for uniformity across all the platforms."""
    df1 = pd.read_csv(file_name, index_col=None, usecols=['username', 'text'])
    df1.rename(columns={'username': 'competitor_name'}, inplace=True)
    df1['platform'] = platform_name.lower()
    df1 = df1[['competitor_name', 'platform', 'text']]
    return df1


def combine_insta_data(filename):
    """Read insta data into a dataframe and rename the columns for uniformity across all the platforms."""
    df1 = pd.read_csv(file_name, index_col=None, usecols=['owner_username', 'caption'])
    df1.rename(columns={'owner_username': 'competitor_name', 'caption': 'text'}, inplace=True)
    df1['platform'] = platform_name.lower()
    df1 = df1[['competitor_name', 'platform', 'text']]
    return df1


def combine_twitter_data(filename):
    """Read twitter data into a dataframe and rename the columns for uniformity across all the platforms."""
    compt_name = re.search(r'Twitter\\(.*?).csv', os.path.join(root, file)).group(1)
    df1 = pd.read_csv(file_name, index_col=None, usecols=['text'])
    # df1.rename(columns = {'owner_username': 'competitor_name', 'caption': 'text'}, inplace=True)
    df1['platform'] = platform_name.lower()
    df1['competitor_name'] = compt_name.lower()
    df1 = df1[['competitor_name', 'platform', 'text']]
    return df1


def combine_website_data(filename):
    """Read website data into a dataframe and rename the columns for uniformity across all the platforms."""
    content2 = []
    compt_name = re.search(r'Website\\(.*?).txt', os.path.join(root, file)).group(1)
    platform = platform_name.lower()

    content2.append(compt_name)
    content2.append(platform)

    with open(filename, encoding='utf8') as f:
        data = f.read().replace('\n', '')
        content2.append(data)

    df_local = pd.DataFrame([content2], columns=['competitor_name', 'platform', 'text'])
    return df_local


directory = "Data Fetching Module/data"

data_frame_comp = pd.DataFrame()
data_frame_influencer = pd.DataFrame()
content = []
content_influencer = []

# Loops through the root data folder containing data from each platform in a .csv file
for root, subdirectories, files in os.walk(directory):
    for file in files:
        file_name = os.path.join(root, file)
        # Using RegEx to pick the subdirectory of platform
        platform_name = re.search(r'data\\(.*?)\\', os.path.join(root, file)).group(1)
        # print(file_name, platform_name)

        if platform_name.lower() == 'facebook':
            df1 = combine_fb_data(file_name)
            content.append(df1)

        elif platform_name.lower() == 'instagram':
            df1 = combine_insta_data(file_name)
            content.append(df1)
        elif platform_name.lower() == 'twitter':
            df1 = combine_twitter_data(file_name)
            content.append(df1)

        elif platform_name.lower() == 'website':
            df1 = combine_website_data(file_name)
            content.append(df1)
        else:
            df_inf = combine_insta_data(file_name)
            df_inf.rename(columns={'competitor_name': 'influencer_username'}, inplace=True)
            content_influencer.append(df_inf)

# Concatening all the data of competitors and influencers into dataframes
data_frame_comp = pd.concat(content)
data_frame_influencer = pd.concat(content_influencer)

data_comp = data_frame_comp.copy()

# Renaming the username values from different platforms into a single unified name.
data_comp.competitor_name.replace('motionnutrition_twitter_posts', 'motionnutrition', inplace=True)
data_comp.competitor_name.replace('neat_nutrition_twitter_posts', 'neat_nutrition', inplace=True)
data_comp.competitor_name.replace('theneurohacker_twitter_posts', 'neurohacker', inplace=True)
data_comp.competitor_name.replace('formnutrition_twitter_posts', 'formnutrition', inplace=True)
data_comp.competitor_name.replace('medterracbd_twitter_posts', 'medterra.international', inplace=True)
data_comp.competitor_name.replace('liveinnermost_twitter_posts', 'liveinnermost', inplace=True)
data_comp.competitor_name.replace('bulkofficial_twitter_posts', 'bulk', inplace=True)
data_comp.competitor_name.replace('puresportcbd_twitter_posts', 'puresport', inplace=True)
data_comp.competitor_name.replace('thenue_co_twitter_posts', 'thenue_co', inplace=True)
data_comp.competitor_name.replace('Bulk_website_text', 'bulk', inplace=True)
data_comp.competitor_name.replace('Neat Nutrition', 'neat_nutrition', inplace=True)
data_comp.competitor_name.replace('The Nue Co.', 'thenue_co', inplace=True)
data_comp.competitor_name.replace('Indi Supplements', 'indisupplements', inplace=True)
data_comp.competitor_name.replace('Medterra International', 'medterra.international', inplace=True)
data_comp.competitor_name.replace('Neurohacker Collective', 'neurohacker', inplace=True)
data_comp.competitor_name.replace('Motion Nutrition', 'motionnutrition', inplace=True)
data_comp.competitor_name.replace('Form Nutrition', 'formnutrition', inplace=True)
data_comp.competitor_name.replace('Form Nutrition_website_text', 'formnutrition', inplace=True)
data_comp.competitor_name.replace('Neat Nutrition_website_text', 'neat_nutrition', inplace=True)
data_comp.competitor_name.replace('The Nue co_website_text', 'thenue_co', inplace=True)
data_comp.competitor_name.replace('INDI MIND_website_text', 'indisupplements', inplace=True)
data_comp.competitor_name.replace('Live innermost_website_text', 'liveinnermost', inplace=True)
data_comp.competitor_name.replace('Medterra_website_text', 'medterra.international', inplace=True)
data_comp.competitor_name.replace('Puresport_website_text', 'puresport', inplace=True)

data_comp.to_csv('data_competitors.csv', index=False)  # output data competitors df to csv file
data_frame_influencer.to_csv('data_influencer.csv', index=False)  # output data of influencer df to csv file
