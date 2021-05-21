import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle as pkl
from sqlalchemy import create_engine
import re


app = Flask(__name__)

def tokenize(text):
    """
    Simple tokenizer we'll use in grid search to see if it's better than
    CountVectorizer's default tokenizer.

    1. Normalize: Strip punctuation and convert to lower
    2. Tokenize: Split message into individual words
    3. Lemmatize: Reduce words to their root, using verb part of speech.
    """
    punct = re.compile('[^A-Za-z0-9]')
    norm = punct.sub(' ', text.lower())

    tokens = [x for x in word_tokenize(norm) if x not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    lemms = [lemmatizer.lemmatize(w, 'v') for w in tokens]

    return lemms

def sort_lists(val_list, label_list, sort_order, top_n):
    """
    Designed to take two lists: labels and values.
    Zips them together, sorts by values, then outputs two lists with matching indices.
    """
    zipped = zip(val_list, label_list)
    ordered = sorted(zipped, reverse = True if sort_order == 'descending' else False)
    tupled = zip(*ordered)
    vals, labels = [list(tuple) for tuple in tupled]

    return vals[0:top_n], labels[0:top_n]


# load data
engine = create_engine('sqlite:///../data/weatheralerts.db')
df = pd.read_sql_table('messages', engine)

# load model
model = pkl.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    labels = [x for x in df.columns if x not in ['id','message','original','genre']]
    label_means = [df[x].mean() for x in labels]

    top_vals, top_labels = sort_lists(label_means, labels, 'descending', 6)
    bot_vals, bot_labels = sort_lists(label_means, labels, 'ascending',6)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {'data':[
            Bar(
                x = top_labels,
                y = top_vals,
                marker_color = 'green'
            )
        ],

        'layout':{
            'title':'Top 6 Categories Represented in Message Set',
            'yaxis':{
                'title': '% of Messages Classified'
            },
            'xaxis': {
                'title': 'Alert Category'
            }
            }
        },

        {'data':[
            Bar(
                x = bot_labels,
                y = bot_vals,
                marker_color = 'red'
            )
        ],
        'layout':{
            'title':'Bottom 6 Categories Represented in Message Set',
            'yaxis':{
                'title':'% of Messages Classified'
            },
            'xaxis':{
                'title': 'Alert Category'
            }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
