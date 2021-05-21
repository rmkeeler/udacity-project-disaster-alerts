import pandas as pd
from sqlalchemy import create_engine
import sys, getopt #let me parse command line arguments

"""
This script as-is is designed specifically to work with two .csv files provided
as part of Udacity Data Scientist Project "Disaster Relief Pipelines."

It imports them using pandas and then applies specific cleaning operations to them
to prepare them for the machine learning step to follow.

Final step is saving

If other datasets are used, modification to this file may be necessary.
"""

def parseargs(argv):
    """
    Parse command line arguments -l and -f (labels.csv and features.csv) and
    then load the datasets specified into pandas dataframes.
    """
    try:
        # Import and parse command line arguments using getopt.getopt
        # First argument is the list of args provided in command prompt (sys.argv)
        # Second is list of short parameters/options (colon after letter says parameter needs a value in cmd)
        # Below, I accept three params: h for help, f for features csv, l for labels csv
        # Third optional arg is a list of long params (only ones that require vals)
        # Gives user the option to include --features=, if that's more comfortable
        # Short args preceded by '-' in cmd. Long ones by '--'
        opts, args = getopt.getopt(argv, 'hf:l:', ['features=','labels='])
    except:
        print('file.py -f <features.csv> -l <labels.csv>')
        sys.exit()

    if '-f' and '-l' not in sys.argv[1:]:
        raise Exception('cmd arguments required:\nfile.py -f <features.csv> -l <labels.csv>')

    for opt, arg in opts:
        print('Opt: {}'.format(opt))
        print('Arg: {}'.format(arg))
        if opt == '-h':
            print('file.py -f <features.csv> -l <labels.csv>')
        elif opt in ('-f','--features'):
            feats = pd.read_csv(arg)
        elif opt in ('-l', '--labels'):
            labs = pd.read_csv(arg)
        else:
            print('file.py -f <features.csv> -l <labels.csv>')
            sys.exit()
    return feats, labs

def cleandata(feats, labs):
    """
    1. Merge feats and labs
    2. Split "categories" into separate columns
    3. Convert category values to 0 or 1
    4. Replace categories column with the new multitude of columns
    5. Remove duplicates
    6. Replace nonbinary values in labels with 1
    """
    # STEP 1
    merged = feats.merge(labs, how = 'left', on = 'id')

    # STEP 2
    split = merged.categories.str.split(';', expand = True)
    split.columns = split.iloc[0].apply(lambda x: x.split('-')[0]).values

    # STEP 3
    for column in split.columns:
        split[column] = split[column].apply(lambda x: int(x[-1]))

    # STEP 3.5
    #labels = [x for x in split.columns]
    #replace = {col: {2:1} for col in labels}
    #split.replace(replace, inplace = True)
    split[~split.isin([0,1])] = 1

    # STEP 4
    merged.drop(columns = ['categories'], inplace = True)
    cleandata = pd.concat([merged, split], axis = 1)

    # STEP 5
    'Duplicate rows found: {}'.format(str(cleandata.duplicated().sum()))
    cleandata.drop_duplicates(inplace = True)
    'Duplicate rows after drop: {}'.format(str(cleandata.duplicated().sum()))

    return cleandata

def loaddata(cleandata, db_name):
    """
    Load cleaned data into a sqlalchemy database file.
    1. Create Engine
    2. Save df to 'messages' table in the database file
    """
    engine = create_engine('sqlite:///data/' + db_name)
    cleandata.to_sql('messages', engine, index = False, if_exists = 'replace')

if __name__ == '__main__':
    features, labels = parseargs(sys.argv[1:])
    cleaned_data = cleandata(features, labels)
    loaddata(cleaned_data, 'weatheralerts.db')
