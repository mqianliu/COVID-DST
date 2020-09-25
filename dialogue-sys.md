# COVID Dialogue System

## Overview
We aim to build a dialogue system that can answer COVID-19 related question from patients automatically. Briefly, we build the system with following steps:
<!-- - **Natural Language Understanding (NLU)**: parse patient's utterance into predefined slots -->
- **Dialogue State Tracking (DST)**: extract patients' INTENTION (their demand) and/or INFORM (their conditions) and encode them into a set of dialogue states till the current turn.
- **Policy Learning**: determine the next system ACTION and/or contents to QUERY from external services based on current dialogue states.
- **Natural Language Generation (NLG)**: generate system response based on ACTION and/or returned QUERY results.

Main challenges lie in the Covid Dialogue System:
- **Reliability/robustness**: a medical dialogue system needs to track the states correctly and act correspondingly.
- **Tracking in time span**: the development of patient's condition (e.g., symptoms, medication) matters and it's non-trivial to track & use. 
- **Service**: how to acquire supporting services, and how to let dialogue system interact with them.
- **Response quality**: generated responses need to be natural, coherent, and consistent with ACTION.
- **Efficiency**: inference time complexity needs to be small.
- **Data insufficiency**: large medical dialogue datasets are typically hard to obtain.

## Building Covid Dialogue System
Given that current CovidDialog dataset is a corpus without semantic labels, we might have few choices if we directly build a task-oriented system from it. For instance, an unstructured [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator) considered the task of best **response selection**, which might not be practical for real-life scenarios. Current mainstream datasets for DST, such as [MultiWoz](https://github.com/budzianowski/multiwoz) and [DSTC 8](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), use structured semantic labels to keep track of dialogue states that are vital for system responses. Hence, our plan is to further annotate the CovidDialog dataset with labels at each turn.

Specifically, our plan is to build up the system with following steps:
### 1. Construct a CovidDST dataset from the CovidDialog dataset.
DST is the core component in a dialogue system. It aims to track the dialogue states from the first to current turns. Please refer to [README](README.md) for more details.
<!-- keeping track of dialogue states is vital for the system to decide what ACTION to take and inquire the external knowledge. -->

<!-- We define the **dialogue state** as a set of **slots** and their corresponding **values**, i.e., the slot-value pair. We keep track of three kinds of slots, namely the INTENTION and INFORM in patients' utterance and ACTION in doctors' utterance. We list some of slots as follows:
<!-- Note that we treat the description as a separate turn and track the INTENTION and INFORM slots for it. -->
<!-- - INTENTION
    - Advise
    - Ask for likelihood of COVID-19 infection
    - Ask for likelihood of cure
    - Ask for likelihood of death
    - Other questions
- INFORM
    - Symptoms
        - Cough
        - Temperature
        - Breath difficulty
        - Vomit
        - Mood
    - Exposure
        - Travel history
        - Contact (with others)
    - Medical condition
    - Medication
    - Physical condition
        - Age
        - Weight
        - Smoking
        - Alcoholic
- ACTION
    - QUERY
    - Advise 
    - Ask for more information -->

    
<!-- In order to track the development of conditions, we further record a **time value** for each value (which is optional), and each slot can have multiple values. We introduce a **< EOV > token** to indicate the end of values for a specific slot.

We devise two types of slots. Some slots have limited predefined values. These slots can be filled via classification. For other slots, values are not provided and they need to be extracted from the utterance. Most slots for INTENTION and ACTION have only two values: YES/NO. 

We show an exemplar consultation in the English CovidDST constructed from CovidDialog dataset as follows:

**Description**

I have cough with no travel history. Is this a symptom of Covid-19?<br>
**INTENTION:** (Ask for likelihood of COVID-19 infection, **YES**). <br>
**INFORM:** (travel history, -, **NO**), (cough, -, **YES**). 

**Dialogue**

**Patient:** Hello doctor, I get a cough for the last few days, which is heavy during night times. No raise in temperature but feeling tired with no travel history. No contact with any Covid-19 persons. It has been four to five days and has drunk a lot of Benadryl and took Paracetamol too. Doctors have shut the OP so do not know what to do? Please help.<br>
**Doctor:** Hello, I understand your concern. I just have a few more questions. Does your cough has phlegm? Any other symptoms like difficulty breathing? Any other medical condition such as asthma, hypertension? Are you a smoker? Alcoholic beverage drinker?<br>
**INTENTION:** (Ask for likelihood of COVID-19 infection, YES), (advise, **YES**). <br>
**INFORM:** (travel history, -, NO), (cough, **last few days**, **heavy during night times**), (temperature, -, **NO**), (tiredness, -, **YES**), (contact, -, **NO**), (medication, **four to five days, drunk a lot of Benadryl and took Paracetamol**). <br>
**ACTION:** (QUERY, **YES**), (ask for more information, **YES**)

**Patient:**
Thank you doctor, I have phlegm but not a lot. A tiny amount comes out most of the time. I have no difficulty in breathing. No medical conditions and not a smoker nor a drinker.<br>
**Doctor:**
Hi, I would recommend you take n-acetylcysteine 200 mg powder dissolved in water three times a day. You may also nebulize using PNSS (saline nebulizer) three times a day. This will help the phlegm to come out. I would also recommend you take vitamin C 500 mg and zinc to boost your immune system. If symptoms persist, worsen or new onset of symptoms has been noted, further consult is advised.<br>
**INTENTION:** (Ask for likelihood of COVID-19 infection, YES), (Advise, YES). <br>
**INFORM:** (travel history, -, NO), (cough, last few days, heavy during night times), (temperature, -, NO), (tiredness, -, YES), (contact, -, NO), (medication, four to five days, drunk a lot of Benadryl and took Paracetamol), (phlegm, -, **not a lot**), (breath difficulty, -, **NO**), (medical condition, -, **NO**), (smoking, -, **NO**), (alcoholic, -, **NO**). <br>
**ACTION:** (QUERY, **YES**), (ask for more information, **NO**), (advise, **YES**) --> 

<!-- Note that we omit the < EOV > token here. -->
### 2. (Optional) Acquire and integrate the supporting services into the system.

### 3. Build and train a dialogue state tracker.
A dialogue state tracker requires to understand the dialogue and track the states in patients' utterance. 

The tracker first encodes the dialogue history up till current turn into a latent space. Then, we use a **state generator** to generate the states for each slot. (Optionally, we can add a slot gate before the generator to filter out irrelevant slots.)

### 4. Build and train a policy maker. 
Given the dialogue states, the policy maker aims to produce system actions for the next step. 

The policy maker first needs to decide whether to inquire the external services. We need to devise rules for the interaction between dialogue system and services, e.g., how to send the query to services and how to store the returned results. Then, the policy maker chooses which ACTION to take.

### 5. Build and train a response generator.
The response generator aims to generate responses based on patients' utterance, dialogue states, system action and query results. 

Language generation model typically generates from scratch. In our dialogue system, we may need to design a copy mechanism that incorporates query results into the response. 

<!-- ## Limitations
- Constructing CovidDST dataset requires tedious labour work. -->


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

    @inproceedings{wu-etal-2019-transferable,
      title = "Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
      author = "Wu, Chien-Sheng  and
      Madotto, Andrea  and
      Hosseini-Asl, Ehsan  and
      Xiong, Caiming  and
      Socher, Richard  and
      Fung, Pascale",
      booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
      month = jul,
      year = "2019",
      address = "Florence, Italy",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/P19-1078",
      doi = "10.18653/v1/P19-1078",
      pages = "808--819"
    }  
