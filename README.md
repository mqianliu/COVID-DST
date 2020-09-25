# COVID-DST Dataset
## Overview
Dialogue State Tracking (DST) is the core component in a [dialogue system](dialogue-sys.md). Compared with vanilla encoder-decoder models, DST typically provides more robustness and expandability for the system via dialogue management.

In this work, we construct COVID-DST dataset from the CovidDialogue dataset that contains patient-doctor conversations about COVID-19 and other related pneumonia. This work is the fundamental part in constructing COVID dialogue system. 

COVID-DST dataset is constructed in the file:
```
covid-dst-en.json
```

## Dataset structure  

### Ontology
We define the **dialogue state** as a set of **slots** and their corresponding **values**, i.e., the slot-value pair. 
The slots and values consist of the ontology of the dialogue system. Slots track patients' description on their conditions and we define them in a hierarchical manner. 
<!-- Nonetheless, COVID-DST is still a single-domain problem, since each slot is unique and we do not consider the "domain" level.   -->

### Slots for patients
<!-- We define two types of slots, namely the **single-value slots** and **multiple-value slots**.  -->
Values in most slots are either "**yes/no**", or are **extracted/summarized** from the utterances. All slots are listed as follows (in lowercase). 
<!-- Note that multiple-value slots are marked as "(multi)". -->

<!-- We devise two types of slots. Some slots have limited predefined values. These slots can be filled via classification. For other slots, values are not provided and they need to be extracted from the utterance. Most slots for INTENTION have only two values: YES/NO.  -->

<!-- In order to track the development of conditions, we further record a **time value** for each value (which is optional), and each slot can have multiple values. We introduce a **\<EOV> token** to indicate the end of values for a specific slot. -->

<!-- We define **39 slots in total**.  -->


<!-- - INTENTION
    - advise
    - covid infection (likelihood)
    - curable
    - death
    - covid-info
    - other-question -->

- Symptoms
    - Lungs
        - cough
        - phlegm
        - breath
        - chest: include chest pain and chest discomfort
    - Upper respiratory tract
        - runny nose
        - throat: include sore throat, throat discomfort
    - Systemic
        - fever
        - chills
        - pain/aches
        - fatigue/weakness
    - Central
        - headache/lightheaded
        - mood
    - gastric
    - other symptoms: recording other symtoms
- medication
- Diagnosis (medical condition)
    - pneumonia
    - asthma
    - diabetes
    - other diagnosis
- Exposure  
    - travel: travel history
    - exposure: possible to be exposured to COVID-19
- Physical condition
    - age 
    - smoking
    - other-phy-con


### Slots for doctors (Dialogue act)
In addition to the ontology, we also need to define a set of dialogue acts to keep track of how doctors respond to the patients. We define two types of dialogue acts. One type of acts can have multiple values, while the other type can only take two values (yes/no). 

Note that the REQUEST act only take the slots defined for patients as values. For example, the doctor asks patients for more detailed information on cough and breath, which is represented as (REQUEST, [cough, breath]). Dialogue acts are listed as follows:
<!-- - QUERY (access the external service) -->
- REQUEST: take the slots as values
- action: suggest what actions to take
- prescription
- diagnose
- Checking 
    - chest X-ray/CT 
    - other-checking
- reqmore (current info is insufficient)
- answer
- knowledge base: track external knowledge for the dialogue system.

### Annotated sample
We show an exemplar consultation in the English CovidDST (ID=1) constructed from CovidDialog dataset as follows:

**Description**: I have cough with no travel history. Is this a symptom of Covid-19?

**Dialogue**

**Patient:** Hello doctor, I get a cough for the last few days, which is heavy during night times. No raise in temperature but feeling tired with no travel history. No contact with any Covid-19 persons. It has been four to five days and has drunk a lot of Benadryl and took Paracetamol too. Doctors have shut the OP so do not know what to do? Please help.<br>
**Doctor:** Hello, I understand your concern. I just have a few more questions. Does your cough has phlegm? Any other symptoms like difficulty breathing? Any other medical condition such as asthma, hypertension? Are you a smoker? Alcoholic beverage drinker?<br>
**SLOT:** (travel, **no**), (cough, **yes**), (fever, **no**), (fatigue, **yes**), (community, **no**), (medication, **[Benadryl, Paracetamol]**). <br>
**ACT:** (REQUEST, **[cough, breath, asthma, blood, smoking]**), (reqmore, **true**)

**Patient:**
Thank you doctor, I have phlegm but not a lot. A tiny amount comes out most of the time. I have no difficulty in breathing. No medical conditions and not a smoker nor a drinker.<br>
**Doctor:**
Hi, I would recommend you take n-acetylcysteine 200 mg powder dissolved in water three times a day. You may also nebulize using PNSS (saline nebulizer) three times a day. This will help the phlegm to come out. I would also recommend you take vitamin C 500 mg and zinc to boost your immune system. If symptoms persist, worsen or new onset of symptoms has been noted, further consult is advised.<br>
**SLOT:** (travel, no), (cough, yes), (fever, no), (fatigue, yes), (community, no), (medication, [Benadryl, Paracetamol]), (breath, **no**), (asthma, **no**), (smoking, **no**). <br>
**ACT:** (prescription, **[n-acetylcysteine, nebulize using PNSS (saline nebulizer), vitamin C 500 mg and zinc]**)

<!-- Note that we omit the \<EOV> token here. -->

## Dataset statistics

COVID-DST contains 603 conversations and 1232 utterances. Most of dialogues contain one turn for English dataset. The average, maximum, and minimum number of utterances in a conversation is 2.0, 17, and 2 respectively. 
### Slot count
**There are 33 slots in total. 24 slots for patient and 9 slots (dialogue acts) for doctors.** We count the number of annotated samples for each slots and dialogue acts.


| Slots | #Samples | Slots | #Samples | Slots | #Samples |
|  ---- | ----  |  ----  | ----  | ----  | ----  |
| Action         | 359 | Diagnose      | 94 | Gastric | 38 
| Answer         | 327 | KB            | 86 | Fatigue | 37
| Cough          | 160 | Otherphycon   | 72 | Phlegm  | 34
| Medication     | 155 | Headache      | 68 | CT/Xray | 33
| Pneumonia      | 126 | Exposure      | 67 | Mood    | 31
| Fever          | 124 | Chest         | 65 | Diabetes| 25
| Prescription   | 124 | Breath        | 63 | Chills  | 20
| Throat         | 118 | Travel        | 57 | Smoking | 20
| Othersym       | 115 | Pain          | 53 | REQUEST | 20
| Otherdiagnosis | 115 | Runnynose     | 44 | Asthma  | 16
| Age            | 97  | Otherchecking | 42 | Reqmore | 13

## Baseline model
Please refer to the folder `.\baseline` for more details.

## Evaluation
We plan to adopt the **joint goal accuracy** - This is the average accuracy of predicting all slot assignments for a turn correctly. For non-categorical slots a fuzzy matching score is used to reward partial matches with the ground truth following [DSTC8](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/master/dstc8.md).

## Challenges for the datasets
- The samples for slots are **imbalanced and insufficient**.
- Hard to map all important descriptions/points in the patients' utterances to the slots.
- Hard to define a set of dialogue acts to fully describe how doctors respond.
- Hard to model the **conditioning realations,** e.g., get tested if symptomatic. 

## Issues
### Major issues
- The values in the "action" slot should be unified.
    - e.g., drink fluids, stay home, self-quarantine, call doctor, etc.
- There are more details to track, e.g., the details of symptoms, the dose of medication.
- Multiple values need to be combined into one in a proper way.
- Some slots could be redundant.
### Minor issues
- Typos or wrong conjunction (e.g., HiIam).
- Some samples are in other languages (ID = 270, 322, 564). 

## References

    @article{ju2020CovidDialog,
      title={CovidDialog: Medical Dialogue Datasets about COVID-19},
      author={Ju, Zeqian and Chakravorty, Subrato and He, Xuehai and Chen, Shu and Yang, Xingyi and Xie, Pengtao},
      journal={ https://github.com/UCSD-AI4H/COVID-Dialogue}, 
      year={2020}
    }

    @article{10.1145/3166054.3166058,
      author = {Chen, Hongshen and Liu, Xiaorui and Yin, Dawei and Tang, Jiliang},
      title = {A Survey on Dialogue Systems: Recent Advances and New Frontiers},
      year = {2017},
      issue_date = {November 2017},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      volume = {19},
      number = {2},
      issn = {1931-0145},
      url = {https://doi.org/10.1145/3166054.3166058},
      doi = {10.1145/3166054.3166058},
      journal = {SIGKDD Explor. Newsl.},
      month = nov,
      pages = {25–35},
      numpages = {11}
    }
