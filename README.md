# Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2023.

## Dataset v1.0

 - 22.5.2023: Release of final dataset v1.0 (train and val), updated evaluation script and baselines
 - 17.4.2023: Release of preliminary dataset v0.9, evaluation script, GPT-baseline

### Baselines

As baselines, we provide:
 - A script that can run masked LMs and causal LMs from Huggingface in the baseline.py, use these to generate entity surface forms, and use a Wikidata API for entity disambiguation.
 - A GPT-3 baseline that directly predicts Wikidata identifiers.

Running instructions for the Huggingface baselines:
 - For BERT

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0```

 - For OPT-1.3b

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0```

 - Run instructions GPT-3 baseline:

 ```python baseline-GPT3-IDs-directly.py" --input data/val.jsonl --output data/testrun-GPT3.jsonl -k YOUR_OPENAI_KEY_HERE```
 
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

```text
| Relation                    \|Train\| \|Val\| \|Test\| Has-None |
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
