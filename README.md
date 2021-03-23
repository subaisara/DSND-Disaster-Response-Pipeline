# Disaster Response Pipeline Project



### Motivation
I  used Data Engineering skills to analyze disaster data. The classifier model is built using ETL process (Extract, Transform and Load), NLP (Natural Language Processing) and Machine Learning pipeline to classify disaster messages. This project also has a web app enables disaster response agency employee to input a new message and get classification results in multiple categories. It is useful to detect what message actually needs attention during a disaster.


### Project Components
There are three main components in this project:

1. ETL Pipeline <br/>
File `data/process_data.py` contains data cleaning pipeline that:
    - Loads the messages and categories dataset
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

2. ML Pipeline
File `models/train_classifier.py` contains machine learning pipeline that:

Loads data from the SQLite database
Splits the data into training and testing sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using LinearSVC
Outputs result on the test set
Exports the final model as a pickle file

3. Flask Web App
    This will start the web application.
    
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
