import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_text_sentiment(input_text):

    sentiment_analyzer = SentimentIntensityAnalyzer()

    sentiment_score = sentiment_analyzer.polarity_scores(input_text)

    return sentiment_score


def parse_file_to_dict(file_path):
    mapping_df = pd.read_excel(file_path, sheet_name='map')

    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: x.split(','))
    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: [keyword.strip() for keyword in x])

    mapping_df['Category'] = mapping_df['Category'].str.strip()
    mapping_df['Style'] = mapping_df['Style'].str.strip()

    mapping_df.set_index('Category', inplace=True)

    category_dict = mapping_df.to_dict(orient='index')

    return category_dict


def get_text_category(input_text, category_dict):

    for category in category_dict.keys():

        category_list = category_dict[category]['Keywords']

        for keyword in category_list:

            if keyword in input_text:

                output_dict = {'Category': category}

                output_dict.update({'Style': category_dict[category]['Style']})

                return output_dict

    return None


def get_sentiment_geometry_from_text(input_text):

    file_path = '/Users/juanorduz/Documents/codinsky_reply/data/category_mapping.xlsx'

    category_dict = parse_file_to_dict(file_path)

    text_sentiment_dict = {'Sentiment': get_text_sentiment(input_text)}

    text_geometry_dict = get_text_category(input_text, category_dict)

    output_dict = text_sentiment_dict

    if text_geometry_dict is not None:
        output_dict.update(text_geometry_dict)

    return output_dict


if __name__ == "__main__":

    input_text = input()

    output_dict = get_sentiment_geometry_from_text(input_text)

    print(output_dict)



