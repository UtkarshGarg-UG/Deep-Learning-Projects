import re


def tokenize(text):
    '''
    desc:
        Tokenize text
    params : 
        text to be tokenized

    Returns : 
        tokenized text
    '''
    
    #pattern = r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+"
    
    return list(text)#re.findall(pattern, text.strip().lower())


