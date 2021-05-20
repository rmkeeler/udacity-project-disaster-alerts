# System packages
import sys

# NLP packages
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Analysis packages
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle as pkl

# Machine Learning packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Get the database from the "data" folder in this project's structure.
    Relevant data must be in a table called "messages" in that database.

    Return X, y and category names.
    X is the single column containing message text. We'll extract features from it.
    y is an array of multiple categories, so this is a multioutput classifier problem.
    y categories take on 1 if category applies to a message, otherwise 0.
    Category names are the names of the categories appearing in y.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name = 'messages', con = engine)

    feature_vars = ['message']
    non_vars = ['id','original','genre']

    target_vars = [x for x in df.columns if x not in feature_vars + non_vars]

    X = df.message.values
    y = df[target_vars].values
    cats = df[target_vars].columns.values

    return X, y, cats


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


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])

    params = {
    'vect__max_features':[None, 5000],
    'tfidf__use_idf':[True, False],
    'clf__estimator':[RandomForestClassifier(), MultinomialNB()]
    }

    cv = GridSearchCV(pipeline, param_grid = params, cv = 2, verbose = 3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Eval model using classification_report().
    Above each eval table, print the name of the output var being evaluated.

    Does nothing but eval the model and print output to console.
    """
    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[1]):
        print('{}'.format(category_names[i]))
        print(classification_report(Y_test[:,i], y_pred[:,i]) + '\n')

def save_model(model, model_filepath):
    """
    Save the cv model to filepath specified in cmd prompt. As pickle.
    """
    pkl.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
