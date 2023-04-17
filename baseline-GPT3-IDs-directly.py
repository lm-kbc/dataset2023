import json
import openai
import ast
from file_io import *
from evaluate import *
import time

def GPT3response(q):
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=q,
                temperature=0,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )    
            response = response.choices[0].text
            response = ast.literal_eval(response)
        except Exception as err:
            print ("Following error occurred in GPT-3 prediction \"{}\". Force stop and run again if error persists. Running again .....".format(err))
            response = None
            time.sleep(10)
    return response
        

if __name__ == '__main__':
    train_filepath = Path("./data/train.jsonl")
    output_filepath = Path("./data/output.jsonl")
    openai.api_key = "INSERT_YOUR_KEY_HERE"
    
    if ("INSERT_YOUR_KEY_HERE" == openai.api_key):
        print ("\n\nEnter your current API key to run the script!!!\n\n")
        exit()

    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    ''' 

    print('Stating probing GPT-3 ................')

    train_df = read_lm_kbc_jsonl_to_df(train_filepath)
    
    # for monetary and test purposes, we take a sample from the dataframe for country-borders-country relation
    train_df = train_df[train_df['Relation'] == 'country-borders-country'].sample(10).reset_index(drop=True)
    print (train_df)

    results = []
    for idx, row in train_df.iterrows():
        prompt = prefix + row["SubjectEntity"] + ", " + row["Relation"] + ", "
        print("Prompt is \"{}\"".format(prompt))
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": GPT3response(prompt),  ## naming prediction IDs directly as objectEntities here 
        }
        results.append(result)

    save_df_to_jsonl(output_filepath, results)

    results = pd.DataFrame(results, columns=['SubjectEntityID', 'SubjectEntity', 'Relation', 'ObjectEntities'])
    train_df = train_df.drop(columns={'ObjectEntities'}).rename(columns={'ObjectEntitiesID': 'ObjectEntities'})

    scores_per_sr_pair = evaluate_per_sr_pair(json.loads(results.to_json(orient='records')), json.loads(train_df.to_json(orient='records')))
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))
    
    print('Finished probing GPT_3 ................')