import pandas as pd
import argparse
import openai
import time
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
    Timeout,
    InvalidRequestError,
)
import os
from tqdm import tqdm
import json
import numpy as np
import ipdb
import pathlib
import re
import tiktoken
import asyncio
import difflib
from split_words import Splitter
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
import torch
import torch.nn.functional as F
from googletrans import Translator
from fuzzywuzzy import fuzz


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


np.random.seed(42)


nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe("language_detector", last=True)


from prompt_template import PROMPT_TEMPLATES
from api_key import API_KEY

CHAT_COMPLETION_MODELS = ["gpt-3.5-turbo", "gpt-4"]
TEXT_COMPLETION_MODELS = ["text-davinci-003"]
COSTS = {
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "text-davinci-003": {"input": 0.00002, "output": 0.00002},
}
ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
}


def read_json(path, lastN=None):
    loaded_lines = []
    with open(path, "r", encoding="utf-8") as f:
        if lastN is None:
            lines = f.readlines()
        else:
            lines = f.readlines()[
                -lastN:
            ]  # TODO fix, doesn't wrk because it's single line json) AD: check how to read last N lines of a JSON (related to num_samples argument)
    for line in lines:
        element = json.loads(line)
        loaded_lines.append(element)
    return loaded_lines


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def detect_language(text):
    maximum = max(len(text), 150)
    doc = nlp_model(text[:maximum])
    detect_language = doc._.language
    return detect_language["language"]


def split_sentences(text):
    # sentences = re.split(r'(?<=[.!?]) +', text)
    # sentences = text.split("\n\n")  # TODO: AD test number of sentences here
    splitter = SentenceSplitter(language="de")
    sentences = splitter.split(text)
    # if sentences are took long, split them again
    sentences = [sentence.rstrip(".") for sentence in sentences]
    # if sentences shorter than 5 words, merge with next sentence
    for idx, sentence in enumerate(sentences):
        if len(sentence.split()) < 5 and idx < len(sentences) - 1:
            sentences[idx + 1] = sentence + " " + sentences[idx + 1]
            sentences[idx] = ""
    sentences = [sentence for sentence in sentences if sentence != ""]

    # # save long sentences to see what's going on
    # long_sents = [sentence for sentence in sentences if len(sentence.split()) > 50]
    # with open("diag_long_sents.txt", "w", encoding="utf-8") as f:
    #     f.write("\n\n".join(long_sents))
    return sentences


def drop_short_text(df, text_col, min_length=100):
    df["text_length"] = df[text_col].apply(lambda x: len(x.split()))
    df = df[df["text_length"] > min_length].drop(columns=["text_length"])

    return df


def replace_html_tags(text):
    def replace_tags(_):
        nonlocal tag_count
        tag_count += 1
        if tag_count % 10 == 0:
            return ". "
        else:
            return " "

    tag_count = 0
    pattern = r"<.*?>"
    result = re.sub(pattern, replace_tags, text)
    return result


def num_tokens_from_string(string, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def compute_cost(input, output, model):
    input_len = num_tokens_from_string(input, model)
    output_len = num_tokens_from_string(output, model)
    return input_len * COSTS[model]["input"] + output_len * COSTS[model]["output"]


def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20, **model_args
            )
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 5 seconds.")
            time.sleep(10)
            continue


# async def batch_chat_completion(
#     messages, model="gpt-3.5-turbo", return_text=True, model_args=None
# ):
#     async_responses = [
#         openai.ChatCompletion.create(
#             model=model,
#             messages=msg,
#             request_timeout=20,
#             **model_args,
#         )
#         for msg in messages
#     ]
#     time.sleep(1)
#     return await asyncio.gather(*async_responses)


def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20, **model_args
            )
            # response = asyncio.run(
            #     batch_chat_completion(messages, model, return_text, model_args)
            # )
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 5 seconds.")
            time.sleep(5)
            continue


def text_completion(
    prompt, model="text-davinci-003", return_text=True, model_args=None
):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.Completion.create(
                model=model, prompt=prompt, request_timeout=20, **model_args
            )
            if return_text:
                return response["choices"][0]["text"].strip()
            return response
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:
            print("Timed out. Waiting for 5 seconds.")
            time.sleep(5)
            continue


def get_extraction_prompt_elements(
    data_type,
    prompt_type,
):
    try:
        data_dict = PROMPT_TEMPLATES[data_type]["extraction"]
    except KeyError:
        raise ValueError("Invalid data_type (should be job, course or cv)")

    try:
        prompt_dict = data_dict[prompt_type]
    except KeyError:
        raise ValueError("Invalid prompt_type, should be skills or wlevels")

    system_prompt = PROMPT_TEMPLATES[data_type]["system"]
    instruction_field = prompt_dict["instruction"]
    shots_field = prompt_dict["shots"]

    return system_prompt, instruction_field, shots_field


def get_matching_prompt_elements(data_type):
    try:
        data_dict = PROMPT_TEMPLATES[data_type]["matching"]
    except KeyError:
        raise ValueError("Invalid data_type (should be job, course or cv)")

    system_prompt = PROMPT_TEMPLATES[data_type]["system"]
    instruction_field = data_dict["instruction"]
    shots_field = data_dict["shots"]

    return system_prompt, instruction_field, shots_field


class OPENAI:
    def __init__(self, args, data, custom_shots_field=None):
        """
        data is a list of dictionaries, each consisting of one sentence and extracted skills
        """
        openai.api_key = args.api_key
        self.args = args
        self.data = data
        self.custom_shots_field = custom_shots_field
        

    def do_prediction(self, task):
        cost = self.run_gpt(task)
        # print("Costs: ", task, cost)
        return self.data, cost

    def run_gpt(self, task):
        if task == "extraction":
            return self.run_gpt_df_extraction()
        elif task == "matching":
            return self.run_gpt_df_matching()

    def run_gpt_df_extraction(self):
        costs = 0
        pattern = r"@@(.*?)##"
        # for idx, sample in enumerate(tqdm(self.data)): TODO REMOVE
        for idx, sample in enumerate(self.data):
            (
                system_prompt,
                instruction_field,
                shots_field,
            ) = get_extraction_prompt_elements(
                self.args.data_type, self.args.prompt_type
            )
            # 1) system prompt
            messages = [{"role": "system", "content": system_prompt}]

            # 2) instruction:
            messages.append(
                {
                    "role": "user",
                    "content": instruction_field,
                }
            )
            

            # 3) shots
            if(self.custom_shots_field is not None):
                ## SPECIFIC SHOTS
                shots_field = self.custom_shots_field

            
            for shot in shots_field[: self.args.shots]:
                sentence = shot.split("\nAnswer:")[0].split(":")[1].strip()
                answer = shot.split("\nAnswer:")[1].strip()
                messages.append({"role": "user", "content": sentence})
                messages.append({"role": "assistant", "content": answer})

            # 4) user input
            messages.append({"role": "user", "content": sample["sentence"]})
            max_tokens = self.args.max_tokens

            prediction = (
                self.run_gpt_sample(messages, max_tokens=max_tokens).lower().strip()
            )
            if self.args.prompt_type == "wlevels":
                # extracted_skills would be the keys and mastery level would be the values
                # keep only the dictionary
                # prediction = prediction.replace("'", '"')
                try:
                    prediction = json.loads(prediction)
                except:
                    print("Error parsing json:", prediction)
                    prediction = {}
                extracted_skills = list(prediction.keys())
                levels = list(prediction.values())
            else:
                extracted_skills = re.findall(pattern, prediction)
            sample["extracted_skills"] = extracted_skills  # AD: removed duplicates
            if self.args.prompt_type == "wlevels":
                sample["extracted_skills_levels"] = levels
            self.data[idx] = sample
            # cost = compute_cost(input_, prediction, self.args.model)
            # costs += cost
            # TODO recompute cost
        return costs

    def run_gpt_df_matching(self):
        costs = 0
        (
            system_prompt,
            instruction_field,
            shots_field,
        ) = get_matching_prompt_elements(self.args.data_type)

        # for idxx, sample in enumerate(tqdm(self.data)): TOTO REMOVE
        for idxx, sample in enumerate(self.data):
            sample["matched_skills"] = {}
            for extracted_skill in sample["extracted_skills"]:
                # 1) system prompt
                messages = [{"role": "system", "content": system_prompt}]

                # 2) instruction:
                messages.append(
                    {
                        "role": "user",
                        "content": instruction_field,
                    }
                )

                # 3) shots

                if(self.custom_shots_field is not None):
                    shots_field = self.custom_shots_field
                for shot in shots_field:
                    sentence = shot.split("\nAnswer:")[0]
                    answer = shot.split("\nAnswer:")[1].strip()
                    messages.append({"role": "user", "content": sentence})
                    messages.append({"role": "assistant", "content": answer})

                # TODO 1.5 having definition or not in the list of candidates ? Here we only prove the name and an example. Yes, should try, but maybe not if there are 10 candidates...
                # update as an argument - like give def or not when doing the matching then ask Marco if it helps or decreases performance

                # 4) user input
                user_input = ""
                options_dict = {
                    letter.upper(): candidate["name+definition"]
                    for letter, candidate in zip(
                        list("abcdefghijklmnopqrstuvwxyz")[
                            : len(sample["skill_candidates"][extracted_skill])
                        ],
                        sample["skill_candidates"][extracted_skill],
                    )
                }
                options_string = " \n".join(
                    letter + ": " + description
                    for letter, description in options_dict.items()
                )
                user_input += f"Sentence: {sample['sentence']} \nSkill: {extracted_skill} \nOptions: {options_string}"

                messages.append({"role": "user", "content": user_input})


                
                # messages_list = [
                #     messages + [{"role": "user", "content": option}]
                prediction = (
                    self.run_gpt_sample(messages, max_tokens=10).lower().strip()
                )

                chosen_letter = prediction[0].upper()
                # TODO match this with the list of candidates, in case no letter was generated! (AD: try to ask it to output first line like "Answer is _")
                # Here the best way is just to change the prompt and ask the model to always output the same template, to make the extraction of the chosen option easier.
                # AD: maybe try JSON or "Answer in _ format" or with specific tags
                # AD: maybe experiment with different params (temperature)
                chosen_option = (
                    options_dict[chosen_letter]
                    if chosen_letter in options_dict
                    else "None"
                )

                for skill_candidate in sample["skill_candidates"][extracted_skill]:
                    if skill_candidate["name+definition"] == chosen_option:
                        sample["matched_skills"][extracted_skill] = skill_candidate
                        break  # stop searching once matched

                self.data[idxx] = sample

                # cost = compute_cost(input_, prediction, self.args.model)
                # costs += cost
        return costs

    def get_num_tokens(self, text):
        encoding = tiktoken.encoding_for_model(self.args.model)
        num_tokens_list = []
        if type(text) == list:
            for item in text:
                num_tokens = len(encoding.encode(item["sentence"]))
                num_tokens_list.append(num_tokens)
            return num_tokens_list
        if type(text) == str:
            num_tokens = len(encoding.encode(text))
            return num_tokens

    def run_gpt_sample(self, messages, max_tokens):
        if self.args.model in CHAT_COMPLETION_MODELS:
            response = chat_completion(
                messages,
                model=self.args.model,
                return_text=True,
                model_args={
                    "temperature": self.args.temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.args.top_p,
                    "frequency_penalty": self.args.frequency_penalty,
                    "presence_penalty": self.args.presence_penalty,
                },
            )

            # get num_tokens of response
            # num_tokens = self.get_num_tokens(response)

        elif self.args.model in TEXT_COMPLETION_MODELS:
            response = text_completion(
                messages,
                model=self.args.model,
                return_text=True,
                model_args={
                    "temperature": self.args.temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.args.top_p,
                    "frequency_penalty": self.args.frequency_penalty,
                    "presence_penalty": self.args.presence_penalty,
                },
            )
            # num_tokens = self.get_num_tokens(response)

        else:
            raise ValueError(f"Model {self.args.model} not supported for evaluation.")

        return response


def concatenate_cols_skillname(row):
    output = row["Type Level 2"]
    output += f": {row['Type Level 3']}" if not pd.isna(row["Type Level 3"]) else ""
    output += f": {row['Type Level 4']}" if not pd.isna(row["Type Level 4"]) else ""
    output += f": {row['Definition']}" if not pd.isna(row["Definition"]) else ""
    return output


# def filter_subwords(extracted_skill, skill_names, splitter):
def filter_subwords(extracted_skill, splitter):
    subwords = []
    for word in extracted_skill.split():
        subwords.extend(list(splitter.split_compound(word)[0][1:]))
    subwords = list(set(subwords))
    subwords = [word for word in subwords if len(word) > 2]
    # matched_elements = []
    # for subword in subwords:
    #     matched_elements.extend(
    #         filter(lambda item: subword in item[1], enumerate(skill_names))
    #     )
    return subwords  # matched_elements


def load_taxonomy(args):
    taxonomy = pd.read_csv(args.taxonomy, sep=",")
    taxonomy = taxonomy.dropna(subset=["Definition", "Type Level 2"])
    taxonomy["name+definition"] = taxonomy.apply(concatenate_cols_skillname, axis=1)
    # taxonomy["unique_id"] = list(range(len(taxonomy)))
    # skill_definitions = list(taxonomy["Definition"].apply(lambda x: x.lower()))
    # skill_names = list(taxonomy["name+definition"].apply(lambda x: x.lower()))

    keep_cols = [
        "unique_id",
        "ElementID",
        # "Dimension",
        "Type Level 1",
        "Type Level 2",
        "Type Level 3",
        "Type Level 4",
        "Example",
        "Definition",
        "name+definition",
    ]
    taxonomy = taxonomy[keep_cols]
    return taxonomy  # , skill_names, skill_definitions


def get_emb_inputs(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return tokens


def find_best_matching_tokens(skill_tokens, sentence_tokens, threshold=95):
    best_matches = []
    best_start_idx = None
    best_end_idx = None

    for i in range(len(sentence_tokens) - len(skill_tokens) + 1):
        score = sum(
            fuzz.ratio(skill_token, sentence_tokens[i + j])
            for j, skill_token in enumerate(skill_tokens)
        )
        if score >= threshold and (best_start_idx is None or score > best_matches):
            best_matches = score
            best_start_idx = i
            best_end_idx = i + len(skill_tokens)

    return best_start_idx, best_end_idx


def get_token_idx(sentence, skill, tokenizer, threshold=95):
    sentence = sentence.lower().strip()
    sentence_tokens = tokenizer.tokenize(sentence)
    skill_tokens = tokenizer.tokenize(skill)

    start_idx, end_idx = find_best_matching_tokens(
        skill_tokens, sentence_tokens, threshold
    )

    if start_idx is None:
        print("String not found in sentence for: ", skill)

    return start_idx, end_idx


def get_embeddings(input_tokens, model):
    with torch.no_grad():
        word_outputs = model(**input_tokens)
        embeddings = word_outputs.last_hidden_state
    return embeddings


def embed_taxonomy(taxonomy, model, tokenizer):
    taxonomy["embeddings"] = taxonomy["name+definition"].apply(
        lambda x: get_embeddings(get_emb_inputs(x, tokenizer), model)[
            :, 0, :
        ]  # get the CLS token
    )
    keep_cols = ["unique_id", "name+definition", "embeddings"]
    embedded_taxonomy = taxonomy[keep_cols]

    return embedded_taxonomy


def get_top_vec_similarity(
    extracted_skill,
    context,
    emb_tax,
    model,
    tokenizer,
    max_candidates=10,
):
    start_idx, end_idx = get_token_idx(context, extracted_skill, tokenizer)
    if start_idx is None and end_idx is None:
        # no idx found for extracted skill so we will take just the embeddings of the skill
        skill_vec = get_embeddings(get_emb_inputs(extracted_skill, tokenizer), model)[
            :, 0, :
        ]  # taking the CLS token since we are comparing against the CLS token of the taxonomy
    else:
        skill_vec = get_embeddings(get_emb_inputs(context, tokenizer), model)[
            :, start_idx:end_idx, :
        ].mean(
            dim=1
        )  # get the contextualized token of skill

    emb_tax["similarity"] = emb_tax["embeddings"].apply(
        lambda x: F.cosine_similarity(x, skill_vec).item()
    )

    cut_off_score = emb_tax.sort_values(by="similarity", ascending=False).iloc[
        max_candidates
    ]["similarity"]
    emb_tax["results"] = emb_tax["similarity"].apply(
        lambda x: True if x >= cut_off_score else False
    )
    return emb_tax

## BASE VERSION
# def select_candidates_from_taxonomy(
#     sample,
#     taxonomy,
#     splitter,
#     model,
#     tokenizer,
#     max_candidates=10,
#     method="rules",
#     emb_tax=None,
# ):
#     assert method in ["rules", "embeddings", "mixed"]
#     sample["skill_candidates"] = {}
#     if len(sample["extracted_skills"]) > 0:
#         for extracted_skill in sample["extracted_skills"]:
#             #print("extracted skill:", extracted_skill)

#             if method == "rules" or method == "mixed":
#                 # print("checking for matches in name+definition")
#                 taxonomy["results"] = taxonomy["name+definition"].str.contains(
#                     extracted_skill, case=False, regex=False
#                 )

#                 if not taxonomy["results"].any():
#                     # print("checking for matches in example")
#                     taxonomy["results"] = taxonomy["Example"].str.contains(
#                         extracted_skill, case=False, regex=False
#                     )

#                 if not taxonomy["results"].any():
#                     # print("checking for matches in subwords")
#                     taxonomy["results"] = False
#                     for subword in filter_subwords(extracted_skill, splitter):
#                         taxonomy["results"] = taxonomy["results"] + taxonomy[
#                             "name+definition"
#                         ].str.contains(subword, case=False, regex=False)

#                 # if not taxonomy["results"].any():
#                 #     if method == "rules":
#                 #         # print("checking for matches in difflib")
#                 #         matching_elements = difflib.get_close_matches(
#                 #             extracted_skill,
#                 #             taxonomy["name+definition"],
#                 #             cutoff=0.4,
#                 #             n=max_candidates,
#                 #         )
#                 #         taxonomy["results"] = taxonomy["name+definition"].isin(
#                 #             matching_elements
#                 #         )

#                 if not taxonomy["results"].any():
#                     print("No candidates found for: ", extracted_skill)

#                 if taxonomy["results"].sum() > 10:
#                     true_indices = taxonomy.index[taxonomy["results"]].tolist()
#                     selected_indices = np.random.choice(true_indices, 10, replace=False)
#                     taxonomy["results"] = False
#                     taxonomy.loc[selected_indices, "results"] = True

#             if method == "embeddings" or method == "mixed":
#                 # print("checking for highest embedding similarity")
#                 emb_tax = get_top_vec_similarity(
#                     extracted_skill,
#                     sample["sentence"],
#                     emb_tax,
#                     model,
#                     tokenizer,
#                     max_candidates,
#                 )
#                 if method == "embeddings":
#                     taxonomy["results"] = emb_tax["results"]
#                 else:
#                     taxonomy["results"] = taxonomy["results"] | emb_tax["results"]
#             # if taxonomy["results"].sum() > 0:
#             #     breakpoint()

#             keep_cols = [
#                 "unique_id",
#                 "Type Level 2",
#                 "name+definition",
#             ]

#             matching_df = taxonomy[taxonomy["results"] == True][keep_cols]

#             sample["skill_candidates"][extracted_skill] = matching_df.to_dict("records")

#     return sample

def select_candidates_from_taxonomy(
    sample,
    taxonomy,
    splitter,
    model,
    tokenizer,
    max_candidates=10,
    method="rules",
    emb_tax=None,
):
    assert method in ["rules", "embeddings", "mixed"]
    sample["skill_candidates"] = {}
    if len(sample["extracted_skills"]) > 0:
        for extracted_skill in sample["extracted_skills"]:
            # print("extracted skill:", extracted_skill)

            # NOTE: this is to handle rule-based/string-matching ways of selecting candidates
            if method == "rules" or method == "mixed":
                taxonomy["results"] = False
                # we will check for exact matches first = 100% skill in name+definition
                taxonomy["match_pct"] = taxonomy["name+definition"].str.contains(
                    extracted_skill, case=False, regex=False
                ).astype(int)
                taxonomy["match_type"] = np.where(
                    taxonomy["match_pct"] == 1, "exact", "none"
                )

                if (taxonomy["match_pct"] == 1).any():
                    if sum(taxonomy["match_pct"] == 1) <= max_candidates:
                        taxonomy["results"] = taxonomy["match_pct"].astype(bool)
                    else:
                        # randomly select max_candidates from the exact matches
                        taxonomy["results"] = taxonomy["match_pct"].sample(
                            n=max_candidates, random_state=42
                        ).astype(bool)
                else:
                    taxonomy["match_pct"] = taxonomy["name"].apply(
                        lambda x: fuzz.token_set_ratio(extracted_skill, x)
                    )
                    # take the top max_candidates
                    taxonomy["results"] = taxonomy["match_pct"].rank(
                        method="first", ascending=False
                    ) <= max_candidates
                    taxonomy["match_type"] = "fuzzy"

            if method == "embeddings" or method == "mixed":
                # print("checking for highest embedding similarity")
                emb_tax = get_top_vec_similarity(
                    extracted_skill,
                    sample["sentence"],
                    emb_tax,
                    model,
                    tokenizer,
                    max_candidates,
                )

                # NOTE: below is to either use the embeddings or a mix of embeddings and rules
                if method == "embeddings":
                    taxonomy["results"] = emb_tax["results"]
                else:
                    taxonomy["results"] = taxonomy["results"] | emb_tax["results"]
            # if taxonomy["results"].sum() > 0:
            #     breakpoint()

            keep_cols = [
                "unique_id",
                "Type Level 2",
                "name",
                "name+definition",
            ]

            matching_df = taxonomy[taxonomy["results"] == True][keep_cols]

            sample["skill_candidates"][extracted_skill] = matching_df.to_dict("records")

    return sample


def add_skill_type(
    df,
):
    pass


def exact_match(
    data,
    tech_certif_lang,
    tech_alternative_names,
    certification_alternative_names,
    data_type,
):
    # Create a dictionary to map alternative names to their corresponding Level 2 values
    synonym_to_tech_mapping = {}
    for _, row in tech_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_tech_mapping[alt_name] = row["Level 2"]

    synonym_to_certif_mapping = {}
    for _, row in certification_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_certif_mapping[alt_name] = row["Level 2"]

    categs = set(tech_certif_lang["Level 1"])
    if data_type == "course":
        categs = categs - set(["Languages"])

    word_sets = [
        set(tech_certif_lang[tech_certif_lang["Level 1"] == categ]["Level 2"])
        for categ in categs
    ]
    # TODO: add separate language processing piece
    for sample in data:
        sentence = sample["sentence"]
        for category, word_set in zip(categs, word_sets):
            # TODO need to exclude the "#" character from being treated as a word boundary in the regular expression pattern! (for C#, same for C++?
            # AD: perhaps list out most common use cases and make an exception for them) -> look in tech_certif_lang.csv
            matching_words = re.findall(
                r"\b(?:"
                + "|".join(re.escape(word) for word in word_set).replace(r"\#", "#")
                + r")\b",
                sentence,
            )
            sample[category] = matching_words

        tech_synonym_set = list(synonym_to_tech_mapping.keys())
        matching_synonyms = re.findall(
            r"\b(?:" + "|".join(re.escape(word) for word in tech_synonym_set) + r")\b",
            sentence,
        )
        matching_tech = [synonym_to_tech_mapping[word] for word in matching_synonyms]
        sample["Technologies"].extend(matching_tech)
        sample["Technologies"] = list(set(sample["Technologies"]))
        sample["Technologies_alternative_names"] = list(set(matching_synonyms))

        certif_synonym_set = list(synonym_to_certif_mapping.keys())
        matching_synonyms = re.findall(
            r"\b(?:"
            + "|".join(re.escape(word) for word in certif_synonym_set)
            + r")\b",
            sentence,
        )
        matching_certif = [
            synonym_to_certif_mapping[word] for word in matching_synonyms
        ]
        sample["Certifications"].extend(matching_certif)
        sample["Certifications"] = list(set(sample["Certifications"]))
        sample["Certification_alternative_names"] = list(set(matching_synonyms))
    return data


def get_lowest_level(row):
    """
    Returns the lowest level of the taxonomy that is not NaN in each
    """
    for level in ["Type Level 4", "Type Level 3", "Type Level 2", "Type Level 1"]:
        value = row[level]
        if not pd.isna(value):
            return value + "_" + level
            # appending level also just in case different levels have the same name


# write something like below that does not work with splice:
def clean_skills_list(skill_name, alternative_names):
    alternative_names = alternative_names.replace("\n", ", ")
    alternative_names = (
        alternative_names.split(":")[1]
        if ":" in alternative_names
        else alternative_names
    )
    alternative_names = re.sub(r"\d+\. ", "", alternative_names)
    alternative_names = alternative_names.split(", ")
    alternative_names = [
        skill for skill in alternative_names if skill != "" and skill_name not in skill
    ]
    # remove if each skill is too long (longer than 10 words)
    alternative_names = [
        skill for skill in alternative_names if len(skill.split()) < 10
    ]
    # remove duplicates
    alternative_names = list(set(alternative_names))
    alternative_names = ", ".join(alternative_names)
    return alternative_names


def clean_text(text):
    text = text.replace("\\n", ". ")
    text = text.strip()
    text = text.replace("..", ".")
    return text


def anonymize_text(text):
    name_pattern = re.compile(r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b")
    phone_pattern = re.compile(r"\b\d{10,12}\b")
    email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

    text = name_pattern.sub("REDACTED_NAME", text)
    text = phone_pattern.sub("REDACTED_PHONE", text)
    text = email_pattern.sub("REDACTED_EMAIL", text)

    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    return text


def remove_level_2(dic):
    if isinstance(dic, dict):
        return {k: remove_level_2(v) for k, v in dic.items() if k != "Type Level 2"}
    elif isinstance(dic, list):
        return [remove_level_2(item) for item in dic]
    else:
        return dic


def remove_namedef(dic):
    if isinstance(dic, dict):
        return {k: remove_namedef(v) for k, v in dic.items() if k != "name+definition"}
    elif isinstance(dic, list):
        return [remove_namedef(item) for item in dic]
    else:
        return dic


def remove_duplicates(dic):
    for key, value in dic.items():
        if isinstance(value, list):
            unique_values = []
            for item in value:
                if item not in unique_values:
                    unique_values.append(item)
            dic[key] = unique_values
        elif isinstance(value, dict):
            remove_duplicates(value)


def translate_text(text, src_lang="de", dest_lang="en"):
    """
    Translates text from one language to another.
    text (str): Text to be translated.
    src_lang (str): Source language.
    dest_lang (str): Target language.
    """
    translator = Translator()
    try:
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
    except:
        print("Time out error. Waiting for 10 seconds...")
        time.sleep(10)
        translator = Translator()
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text
