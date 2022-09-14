import pandas as pd
import numpy as np
import re

# Read the combined competitor and influencer data into dataframes
compiled_comp_data = pd.read_csv("data/data_competitors.csv")
compiled_inf_data = pd.read_csv("data_influencer.csv")

# Remove nulls if there are any
compiled_comp_data.dropna(inplace=True)
compiled_inf_data.dropna(inplace=True)


def remove_emoji(string):
    """This function removes emojis and any non text characters present in a string. It does this using RegEx and finding patterns corresponding to Unicodes."""
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
    return emoji_pattern.sub(r'', string)


def cleanText(text_col):
    """This is the main function that takes each post text from the .csv file and cleans the text corpus"""
    cleaned_item_list = []

    for text in text_col.values:
        clean_text = remove_emoji(text)  # removes emoji from each text
        cleanedText = re.sub(r'[?|!|\'|"|#|+|@|)|(]', r' ', clean_text) # Removes special characters like  @, ?, ! etc.,
        cleanedText = re.sub(r'[*|&|=|$|%|+|@|)|(]', r' ', cleanedText) # Removes special characters like  @, ?, ! etc.,
        cleanedText = re.sub(r'[,|;|:|}|{|-|_|£]', r' ', cleanedText) # Removes special characters like  @, ?, ! etc.,
        pattern = "#(\w+)"
        cleanedText = re.sub(pattern, r'', cleanedText)  # it removes hashtagged words
        cleaned_item_list.append(cleanedText)

    return cleaned_item_list


# Clean the combined competitor data and save it to a dataframe
texts_comp_cleaned = cleanText(compiled_comp_data.text)
compiled_comp_data["cleaned_text"] = texts_comp_cleaned

# Clean the combined influencer data and save it to a dataframe
text_inf_cleaned = cleanText(compiled_inf_data.text)
compiled_inf_data["cleaned_text"] = text_inf_cleaned


def removeCols_changeNames(df):
    if "Unnamed: 0" in df.columns:
        df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.drop(["text"], axis=1, inplace=True)
    df.rename(columns={"cleaned_text": "text"}, inplace=True)


removeCols_changeNames(compiled_comp_data)
removeCols_changeNames(compiled_inf_data)

compiled_comp_data.to_csv(
    "preprocessed_data_comp_without_punc.csv", index=False)  # output final text processed data of competitor into csv file format
compiled_inf_data.to_csv(
    "preprocessed_data_inf_without_punc.csv", index=False)  # output final text processed data of influence into csv file format
