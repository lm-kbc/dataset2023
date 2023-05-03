import json
import openai
import ast
from file_io import *
from evaluate import *
import time

def GPT3response(q):
    response = openai.Completion.create(
        # curie is factor of 10 cheaper than davinci, but correspondingly less performant
        model="text-davinci-003",
        #model = "text-curie-001",
        prompt=q,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )    
    response = response.choices[0].text
    if response[0] == " ":
        response = response[1:]
    try:    
        response = ast.literal_eval(response)
    except:
        response = []
    return response
        

if __name__ == '__main__':
    train_filepath = Path("./val.jsonl")
    output_filepath = Path("./val_predictions.jsonl")
    openai.api_key = "INSERT_YOUR_KEY_HERE"
    
    if ("INSERT_YOUR_KEY_HERE" == openai.api_key):
        print ("\n\nEnter your current API key to run the script!!!\n\n")
        exit()

    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    ''' 

    print('Starting probing GPT-3 ................')

    train_df = read_lm_kbc_jsonl_to_df(train_filepath)
    
    print (train_df)

    results = []
    for idx, row in train_df.iterrows():
        prompt = prefix + row["SubjectEntity"] + ", " + row["Relation"] + ", "
        print("Prompt is \"{}\"".format(prompt))
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntitiesID": GPT3response(prompt),  ## naming with IDs required for current evaluation script 
        }
        results.append(result)

    save_df_to_jsonl(output_filepath, results)

    print('Finished probing GPT_3 ................')