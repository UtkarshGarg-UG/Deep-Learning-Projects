import pandas as pd
import string
from string import digits
from pandarallel import pandarallel
import re
# Initialization
pandarallel.initialize()


def preprocessing(data):
    #clean the data

    data = data.dropna().drop_duplicates()

    #lower and remove quotes
    data['english_sentence'] = data.english_sentence.parallel_apply(lambda x: re.sub("'",'',x).lower())
    data['hindi_sentence'] = data.hindi_sentence.parallel_apply(lambda x: re.sub("'",'',x).lower())

    #remove special charsls
    exclude = set(string.punctuation) # Set of all special characters
    # Remove all the special characters
    data['english_sentence']=data['english_sentence'].parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    data['hindi_sentence']=data['hindi_sentence'].parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


    remove_digits = str.maketrans('', '', digits)
    data['english_sentence']=data['english_sentence'].parallel_apply(lambda x: x.translate(remove_digits))
    data['hindi_sentence']=data['hindi_sentence'].parallel_apply(lambda x: x.translate(remove_digits))

    data['hindi_sentence'] = data['hindi_sentence'].parallel_apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    # Remove extra spaces
    data['english_sentence']=data['english_sentence'].parallel_apply(lambda x: x.strip())
    data['hindi_sentence']=data['hindi_sentence'].parallel_apply(lambda x: x.strip())
    data['english_sentence']=data['english_sentence'].parallel_apply(lambda x: re.sub(" +", " ", x))
    data['hindi_sentence']=data['hindi_sentence'].parallel_apply(lambda x: re.sub(" +", " ", x))

    return data
