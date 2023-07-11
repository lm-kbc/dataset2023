# Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2023.

## Dataset v1.0

 - 22.5.2023: Release of final dataset v1.0 (train and val), updated evaluation script and baselines
 - 17.4.2023: Release of preliminary dataset v0.9, evaluation script, GPT-baseline

### Baselines

As baselines, we provide:
 - A script that can run masked LMs and causal LMs from Huggingface in the baseline.py, use these to generate entity surface forms, and use a Wikidata API for entity disambiguation.
 - A GPT-3 baseline that directly predicts Wikidata identifiers.
 - A GPT-3 baseline that uses Wikidata NED.


Running instructions for the Huggingface baselines:
 - For BERT

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0```

 - For OPT-1.3b

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0```

 - Run instructions GPT-3 baselines:

 ```python baseline-GPT3-IDs-directly.py" --input data/val.jsonl --output data/testrun-GPT3.jsonl -k YOUR_OPENAI_KEY_HERE```
  ```python baseline-GPT3-NED.py" --input data/val.jsonl --output data/testrun-GPT3.jsonl -k YOUR_OPENAI_KEY_HERE```

 
### Evaluation script

Run instructions evaluation script:
  * ```python evaluate.py -p data/val.jsonl -g data/testrun-XYZ.jsonl```

The first parameter hereby indicates the prediction file, the second the ground truth file.

## Relations

```text
1. BandHasMember
2. CityLocatedAtRiver
3. CompanyHasParentOrganisation
4. CompoundHasParts
5. CountryBordersCountry
6. CountryHasOfficialLanguage
7. CountryHasStates
8. FootballerPlaysPosition
9. PersonCauseOfDeath
10. PersonHasAutobiography
11. PersonHasEmployer
12. PersonHasNoblePrize
13. PersonHasNumberOfChildren
14. PersonHasPlaceOfDeath
15. PersonHasProfession
16. PersonHasSpouse
17. PersonPlaysInstrument
18. PersonSpeaksLanguage
19. RiverBasinsCountry
20. SeriesHasNumberOfEpisodes
21. StateBordersState
```


----------------------------------------------------------------

## Dataset Characteristics
Number of unique subject-entities in the data splits.

```text
| Relation                    |Train| |Val| |Test| Has-None       |
|-----------------------------------------------------------------|
| countryBordersCountry         63    63    63       No           |
| countryHasOfficialLanguage    65    65    65       No           |
| stateBordersState             100   100   100      No           |
| CompoundHasParts              66    66    66       No           |
| riverBasinsCountry            100   100   100      No           |
| personHasPlaceOfDeath         100   100   100      Yes          |
| companyHasParentOrganisation  100   100   100      Yes          |
| personSpeaksLanguage          100   100   100      No           |
| personHasProfession           100   100   100      No           |
| personPlaysInstrument         100   100   100      No           |
| seriesHasNumberOfEpisodes     100   100   100      No           |
| personHasNumberOfChildren     100   100   100      No           |
| BandHasMember                 100   100   100      No           |
| CityLocatedAtRiver            100   100   100      No           |
| CountryHasStates              46    46    46       No           |
| FootballerPlaysPosition       100   100   100      No           |
| PersonCauseOfDeath            100   100   100      Yes          |
| PersonHasAutobiography        100   100   100      No           |
| PersonHasEmployer             100   100   100      No           |
| PersonHasNoblePrize           100   100   100      Yes          |
| PersonHasSpouse               100   100   100      No           |                           
```

## Baseline performance

```text
GPT-3 (curie model)
|                                   p      r     f1
|-----------------------------------------------------------------|
| BandHasMember                 0.240  0.000  0.000
| CityLocatedAtRiver            0.000  0.000  0.000
| CompanyHasParentOrganisation  0.000  0.000  0.000
| CompoundHasParts              0.030  0.000  0.000
| CountryBordersCountry         0.392  0.201  0.186
| CountryHasOfficialLanguage    0.046  0.046  0.046
| CountryHasStates              0.554  0.013  0.018
| FootballerPlaysPosition       0.010  0.000  0.000
| PersonCauseOfDeath            0.000  0.000  0.000
| PersonHasAutobiography        0.010  0.000  0.000
| PersonHasEmployer             0.010  0.000  0.000
| PersonHasNoblePrize           0.030  0.000  0.000
| PersonHasNumberOfChildren     0.200  0.200  0.200
| PersonHasPlaceOfDeath         0.000  0.000  0.000
| PersonHasProfession           0.050  0.021  0.028
| PersonHasSpouse               0.000  0.000  0.000
| PersonPlaysInstrument         0.000  0.000  0.000
| PersonSpeaksLanguage          0.340  0.195  0.238
| RiverBasinsCountry            0.496  0.440  0.407
| SeriesHasNumberOfEpisodes     0.200  0.150  0.150
| StateBordersState             0.040  0.000  0.000
| *** Average ***               0.126  0.060  0.061
```

BERT

```text
| p   r   f1
|-----------------------------------------------------------------|
| BandHasMember                 0.460 0.000 0.000
| CityLocatedAtRiver            0.118 0.028 0.025
| CompanyHasParentOrganisation  0.518 0.070 0.062
| CompoundHasParts              0.343 0.134 0.140
| CountryBordersCountry         0.796 0.486 0.567
| CountryHasOfficialLanguage    0.887 0.753 0.775
| CountryHasStates              0.174 0.000 0.000
| FootballerPlaysPosition       0.187 0.507 0.265
| PersonCauseOfDeath            0.022 0.040 0.027
| PersonHasAutobiography        0.570 0.000 0.000
| PersonHasEmployer             0.110 0.000 0.000
| PersonHasNoblePrize           0.000 0.000 0.000
| PersonHasNumberOfChildren     0.000 0.000 0.000
| PersonHasPlaceOfDeath         0.208 0.130 0.108
| PersonHasProfession           0.405 0.008 0.010
| PersonHasSpouse               0.800 0.000 0.000
| PersonPlaysInstrument         0.020 0.006 0.007
| PersonSpeaksLanguage          0.552 0.678 0.569
| RiverBasinsCountry            0.467 0.514 0.415
| SeriesHasNumberOfEpisodes     1.000 0.000 0.000
| StateBordersState             0.088 0.017 0.020
| *** Average ***               0.368 0.161 0.142
```
----------------------------------------------------------------

### YOUR prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntitiesID``: the predicted object entities ID, which should be a list of Wikidata IDs (strings).

You can take a look at the [example prediction file](data/dev.pred.jsonl) to
see how a valid prediction file should look like.

This is how we write our prediction file:

```python
import json

# Fake predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q790", "Q717", "Q30", "Q183"]
    },
    {
        "SubjectEntity": "Eritrea",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q115"]
    },
    {
        "SubjectEntity": "Estonia",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": []
    }

]

fp = "/path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```