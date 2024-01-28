from .feedback_prompt_template import PROMPTS
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    Timeout,
)
from generator import MODELS
import openai
import time
from typing import List, Dict
from tqdm.notebook import tqdm
import pandas as pd
from datasets import load_dataset
import re
from tqdm.notebook import tqdm

import os
API_KEY = os.environ["API_KEY"]
if(API_KEY == ""):
    raise NotImplementedError("You need to enter your OPENAI API key in .env")

PATTERN = r"@@(.*?)##"
SELF_REFINING_TYPES = [
    "BAD-BOUNDS",
    "NO-ANNOT"
]

class SpanExtractor():
    
    def __init__(self,
                 prompt_template: Dict[str, str],
                 model: str="gpt-3.5",
                 model_temp: float=1):
        openai.api_key = API_KEY
        self.model = model
        self.prompt_template = prompt_template
        self.model_temp = model_temp

    def create_prompt_for(self, sample):

        (system_prompt, instruction_field, shots) = list(self.prompt_template["INIT"].values())
        messages = [
            {
                'role': "system",
                "content":system_prompt
            },
            {
                "role": "user",
                "content": instruction_field
            }
        ]
        for shot in shots:
            # [print(x) for x in shot.split("\n")]
            sentence, skill, answer,_ = shot.split("\n")
            messages.append({'role':'user', 'content':sentence + "\n" + skill})
            messages.append({'role':'assistant', 'content': answer})

        messages.append({'role': 'user', 'content': "sentence: " + str(sample["sentence"]) + "\nskill : " + str(sample["skill"])})

        return messages
    
    def self_refining(self,
                      sentence,
                      previous_chat: List[Dict[str, str]],
                      refining_type: str,
                      model: str):
        if(refining_type not in SELF_REFINING_TYPES):
            raise ValueError(f"refining type must be one of {SELF_REFINING_TYPES}")
        
        refine_content = self.prompt_template[refining_type]["content"] + "\n" + "Sentence: " + str(sentence) + "\n"
        new_chat = previous_chat + [{
            'role': 'user',
            'content': refine_content
        }]

        return self.query(new_chat, model)
        


    def query(self, 
              messages:List[Dict[str, str]],
              model: str):
        
        #######
        # print("-"*100)
        # for message in messages:
        #     print(message["content"])
        #######

        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20, temperature=self.model_temp
            )
            return response["choices"][0]["message"]["content"]
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 10 seconds.")
            time.sleep(10)

    def extract_span(self, sample, refining=False):
        if(sample["skill"] == "LABEL NOT PRESENT"):
            return sample["sentence"]
        messages = self.create_prompt_for(sample)
        annot = self.query(messages, MODELS[self.model])

        if(refining):
            messages += [{
                "role": 'assistant',
                "content": annot
            }]
            if("@@" not in annot and "##" not in annot):
                ## not annotation at all
                print(f"> self refinement on : {sample['sentence']} > NO-ANNOT")
                print("before : ", annot)
                annot = self.self_refining(sample['sentence'], messages, "NO-ANNOT", MODELS[self.model])
                print("after : ", annot)
            elif("@@" in annot and "##" not in annot):
                ## wrong annot
                print(f"> self refinement on : {sample['sentence']} > BAD-BOUNDS")
                print("before : ", annot)
                annot = self.self_refining(sample['sentence'], messages, "BAD-BOUNDS", MODELS[self.model])
                print("after : ", annot)
        return annot

        

def annotate_sentences(ds, spanextr, self_refine=False):
    records = ds.to_dict("records")
    annotated_sentences = []
    for _, record in tqdm(list(enumerate(records))):
        annotated_sentences.append(
            spanextr.extract_span(record, refining=self_refine)
        )
    return annotated_sentences


def extract_spans(annot_st):
    return [x.lower() for x in re.findall(PATTERN, annot_st)]


def load_train_gen():
    """
        Return format:
        DataFrame : ["sentence", "skill"] unique
    """
    df_train = pd.read_csv("generated/SKILLSPAN/train_final.csv")
    df_train["skills"] = df_train["skills"].apply(eval)
    df_train = df_train[["sentence", "skills"]]
    df_train.columns = ["sentence", "skill"]
    df_train = df_train.explode("skill")
    return df_train

def load_train_dec():
    """
        Return format:
        DataFrame : ["sentence", "skill"] unique
    """
    return pd.DataFrame(
        load_dataset("jensjorisdecorte/Synthetic-ESCO-skill-sentences")["train"]
    )

def annotate_ds(ds):
    spanextr = SpanExtractor(PROMPTS, "gpt-3.5", model_temp=0.45)
    annot = annotate_sentences(ds, spanextr, self_refine=True)
    ds["annotated_sentence"] = annot
    ds["annot_spans"] = ds["annotated_sentence"].apply(extract_spans)
    return ds

def annotate_clean(sentence: str, annot_spans: List[str]):
    for span in annot_spans:
        sentence = sentence.replace(span, f"@@{span}##")
    return sentence
