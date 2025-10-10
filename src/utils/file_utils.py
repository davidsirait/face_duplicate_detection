import pandas as pd
import unicodedata

def clean_text_unicode(text):
    """
    Remove accents and convert to ASCII using standard library
    """
    # Normalize to NFD (decomposed form)
    nfd = unicodedata.normalize('NFD', text)
    
    # Remove combining characters (accents)
    ascii_text = ''.join(char for char in nfd 
                         if unicodedata.category(char) != 'Mn')
    
    # Keep only ASCII characters
    ascii_text = ascii_text.encode('ascii', 'ignore').decode('ascii')
    
    return ascii_text 

def sanitize_dataframe(df):
    """ 
    Sanitize the name and create title column for the dataframe
    """

    # clean the name column from ascii character
    df['name_clean'] = df['name'].apply(lambda x : clean_text_unicode(x))

    # create the title column for the image title
    df["title"] = df.apply(lambda x: x['name_clean'] + "_" + str(x["image_id"]), axis=1)

    print(f"Data after cleaning:")
    print(df.head())

    return df