## Table of Contents

1. [Installation](#installation)
2. [Usage Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Licensing, Authors and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
Python version used: 3.8.7

Packages used:
1. Pandas 1.2.3
2. Numpy 1.20.2
3. Scikit-Learn 0.24.1
4. NLTK 3.6.2
5. Pickle 0.7.5
6. SQLAlchemy 1.4.15
7. Flask 1.1.2

## Usage Instructions <a name="instructions"></a>

Running data/process_data.py and then models/train_classifier.py in that order will put all necessary files where they need to go. Input datasets can be stored anywhere, but it's recommeded to store them in the data directory.

To run data/process_data.py from the project root:
```
python data/process_data.py -f <messageCSVPath> -l <labelsCSVPath>
```
Where the `-f` argument specifies the path to a CSV file containing the text messages from which the model's features will be derived.

And the `-l` argument specifies the path to a CSV file containng the classifications of the messages in the messages file.

To run models/train_classifier from the project root:
```
python models/train_classifier.py <database_path> <model_save_location>
```

Where `<database_path` is the location of the database you saved when you ran `data/process_data`.

And `<model_save_location>` is the path and filename where you would like to save the model you're training in this step.

At that point, you're basically done. What's left is launching the Flask app, locally. From the project root:
```
python app/run.py
```
The app will launch on a local server at 127.0.0.1:3001, by default. You can change that location by editing app/run.py.

app/run.py also contains the plotly objects and the pandas analysis steps necessary to produce their data. Feel freee to edit, as desired.

## Project Movitation <a name="motivation"></a>

I undertook this project as part of [Udacity's](https://www.udacity.com) Data Scientist nanodegree program. The primary motivations for this project were:

1. Exploring natural language processing and feature extraction
2. Exploring ML pipelining and model optimization
3. Practicing data wrangling via the Extract-Transform-Load process
4. Understanding how to build a data science pipeline that culminates in a public web application

The secondary motivation was the project's simulated goal: Creating a public web application that allows disaster relief workers to input a text message and learn what kind (if any) relief is desired by the sender. Routing social media messages to appropriate disaster relief agencies is the broader goal of the application.

## File Descriptions <a name="files"></a>

1. data/process_data.py: Takes file references in command prompt and performs ETL to clean datasets, merge them into one and load into a local db file.
2. models/train_classifier.py: Extracts data from db file created above, tokenizes messages and builds a predictive model using the resulting tokens. Saves the model to models/classifier.pkl
3. run.py: The Flask app. Creates a local server instances at 127.0.0.1:3001 and produces the web app, there. Also performs the analyses the populate the bar charts in the app.

## Licensing, Authors and Acknowledgements <a name="licensing"></a>

Data collected and provided by [Appen](https://appen.com) for the purpose of a project in [Udacity's](https://www.udacity.com) Data Scientist Nanodegree program.

process_data.py and train_classifier.py provided by Udacity. Functions within them written by me (main() in train_classifier provided by Udacity).

Flask app html and app.py provided by Udacity and altered by me.

Feel free to use the code provided in this repository at your own discretion.
