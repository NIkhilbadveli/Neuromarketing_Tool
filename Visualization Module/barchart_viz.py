import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def show_barchart(comp=True, top_n=30):
    """Generate a barchart and show the image"""
    if comp:
        df = pd.read_csv('competitor_keywords_nltk.csv')
        df.sort_values(by='Count', ascending=False, inplace=True)
        # print(df.head(top_n))
    else:
        df = pd.read_csv('influencer_keywords_nltk.csv')
        df.sort_values(by='Count', ascending=False, inplace=True)

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    # sns.barplot(y='Word', x='Count', data=df[:top_n], ci=100)
    ax.barh(df[:top_n].Word.values, df[:top_n].Count.values)
    if comp:
        plt.title('Competitor Keywords - Top {}'.format(top_n))
    else:
        plt.title('Influencer Keywords - Top {}'.format(top_n))

    # Uncomment this to add values to the bars
    # for i, value in enumerate(df[:top_n].Count.values):
    #     ax.text(value, i, str(value))

    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()


show_barchart(comp=True)
