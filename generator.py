from typing import Dict, List, Any
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from gen_prompt_template import PROMPT_TEMPLATE
from api_key import API_KEY
import openai
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
    Timeout,
    InvalidRequestError,
)
from tqdm.notebook import tqdm
import time
import re
from collections import defaultdict
import json
from math import ceil


class SkillsGenerator():

    def __init__(self, 
                 taxonomy: DataFrame, 
                 taxonomy_is_embedded: bool,
                 combination_dist: npt.ArrayLike,
                 popularity: Dict,
                 emb_model: Any = None,
                 emb_tokenizer: Any = None,
                 reference_df: DataFrame = None,):
        
        if(not taxonomy_is_embedded):
            if("name+definition" not in emb_tax.columns):
                raise ValueError("The taxonomy must contain a 'name+definition' column")
            emb_tax = SkillsGenerator.embedd_df(taxonomy, "name+defintion", emb_model, emb_tokenizer)
        else :
            emb_tax = taxonomy
        
        ## embedded taxonomy
        self.emb_tax = emb_tax 


        ## combination dist
        self.combination_dist = SkillsGenerator.softmax(combination_dist)

        ## popularity measures
        self.popularity = popularity

        ## computing sim matrix
        self.compute_sim_matrix()

        ## Label to Idx and opposite
        self.idx_to_label = {k: v for k, v in enumerate(taxonomy.name.values)}
        self.label_to_idx = {v: k for k, v in enumerate(taxonomy.name.values)}

        ## refenrece df for specific few shots
        self.reference_df = reference_df


    @staticmethod
    def embedd_df(df: DataFrame,
                  key_to_embed:str,
                  model: Any,
                  tokenizer: Any):
        """
            Embedds the entities of the dataframe in column key_to_embed
            using the given model and tokenizer
        """
        df["embeddings"] = df[key_to_embed]\
                    .apply(lambda st : \
                    model(**tokenizer(st, return_tensors="pt", max_length=768, padding=True, truncation=True))\
                    .last_hidden_state[:, 0, :]\
                    )
        return df

    def compute_sim_matrix(self):
        """
            Creates a new field in the class = the pairwise
            similarity matrix between all skill embeddings
        """
        skills_embeddings = torch.cat(list(self.emb_tax["embeddings"].values)).numpy()
        pairwise_sims = cosine_similarity(skills_embeddings, skills_embeddings)
        self.pairwise_sims = pairwise_sims


    def get_combination_for_(self, skill: str,
                             k: int,
                             threshold:float,
                             temperature: float=1,
                             frequency_select: bool=False,
                             temperature_sample_size: float=1,
                             upper_bound_skill_matching:int = None):
        """
            Creates combination of skill to pair with skill

            skill            : the skill we consider
            k                : the number of close neighbor of skill to consider
            threshold        : maximum allowed distance between skill and a candidate
            temperature      : flattening of the frequency distribution of the neighbors
            frequency_select : are the neighbors selected according to their frequency ?

        """
        skill_idx = self.label_to_idx[skill]
        sims_with_skill = self.pairwise_sims[skill_idx, :]
        kNN = (-sims_with_skill).argsort()[1:k+1]
        kNN_skills = [self.idx_to_label[nn] for nn in kNN if self.pairwise_sims[skill_idx, nn] > threshold]
        
        if(len(kNN_skills) == 0):
            return []

        if(upper_bound_skill_matching is None):
            nb_associated_skills = self.get_combination_size(temperature_sample_size) - 1 ## we remove the skill selected first
        else :
            nb_associated_skills = min(self.get_combination_size(temperature_sample_size) - 1, upper_bound_skill_matching) ## we remove the skill selected first

        if(nb_associated_skills == 0):
            return []
        if(frequency_select):
            F = np.array([self.popularity[nn] for nn in kNN_skills])
            F = SkillsGenerator.softmax(-F, T=temperature) ## we get a dist, potentially dumped
            # with frequency dist
            kNN_skills = list(np.random.choice(kNN_skills, size=min(nb_associated_skills, len(kNN_skills)), replace=False, p=F))
                
        else :
            # with uniform dist
            kNN_skills = list(np.random.choice(kNN_skills, size=min(nb_associated_skills, len(kNN_skills)), replace=False))
        return kNN_skills
    

    def get_combination_size(self, T: float):
        """
            Returns a realization of the distribution $\mathcal{N}$, of the combination size dist
            flattened or skewed with the temperature T
        """
        temp_dist = SkillsGenerator.softmax(self.combination_dist, T)
        n = np.random.choice(np.arange(1, len(self.combination_dist) + 1), p=temp_dist) 
        return n


    @staticmethod
    def softmax(X, T=1):
        return np.exp(X / T) / np.sum(np.exp(X / T))

    def stochastic_inf_iter(self, 
                            total_generations=1e7, 
                            threshold=0.0,
                            beam_size=20,
                            temperature_skill=1,
                            temperature_pairing=1,
                            temperature_sample_size=1,
                            frequency_select=True, 
                            upper_bound_skill_matching=None):
        """
            Creates a lazy iterator of combinations of entities in the taxonomy

            parameters:
                - total_generations          : 
                - threshold                  : 
                - beam_size                  : 
                - temperature_skill          :   
                - temperature pairing        : 
                - temperature_sample_size    : 
                - frequency_select           : 
                - upper_bound_skill_matching : 
        """
        all_skills = list(self.label_to_idx.keys())
        F = np.array([self.popularity[sk] for sk in self.label_to_idx.keys()])
        F = SkillsGenerator.softmax(-F, temperature_skill)
        
        for gen in range(total_generations):
            skill = np.random.choice(all_skills, p=F) ## chosing the skill to generate
            combs = [skill] + self.get_combination_for_(skill, 
                                                        threshold=threshold,
                                                        k=beam_size,
                                                        temperature=temperature_pairing,
                                                        frequency_select=frequency_select,
                                                        temperature_sample_size=temperature_sample_size,
                                                        upper_bound_skill_matching=upper_bound_skill_matching) ## get the tuple to generate
            yield combs

    def balanced_iter(self, 
                        skills_to_use='all', 
                        threshold=0.0,
                        beam_size=20,
                        temperature_pairing=1,
                        temperature_sample_size=1,
                        frequency_select=True, 
                        upper_bound_skill_matching=None):
        """
            Creates a list of combinations of entities in the taxonomy

            parameters:
                - total_generations          : 
                - threshold                  : 
                - beam_size                  :   
                - temperature pairing        : 
                - temperature_sample_size    : 
                - frequency_select           : 
                - upper_bound_skill_matching : 
        """
        ## check of skills_to_use
        if((type(skills_to_use) != "int" and skills_to_use != "all")
           or (skills_to_use > len(self.emb_tax.index))):
            raise ValueError("'skills_to_use' must be an int smaller than the number of considered skills or 'all'")
    
        if(skills_to_use == "all"):
            skills_to_use = len(self.emb_tax.index)
        all_skills = list(self.label_to_idx.keys())
        
        skills_to_generate = np.random.choice(all_skills, size=skills_to_use, replace=False)
        
        all_gens = []
        for skill in skills_to_generate:
            all_gens.append([skill] + self.get_combination_for_(skill, 
                                                        threshold=threshold,
                                                        k=beam_size,
                                                        temperature=temperature_pairing,
                                                        frequency_select=frequency_select,
                                                        temperature_sample_size=temperature_sample_size,
                                                        upper_bound_skill_matching=upper_bound_skill_matching) ## get the tuple to generate
            )
        return all_gens
    
    def balanced_nbred_iter(self, 
                            nb_generation=5000, 
                            threshold=0.0,
                            beam_size=20,
                            temperature_pairing=1,
                            temperature_sample_size=1,
                            frequency_select=True, 
                            upper_bound_skill_matching=None, 
                            diverse=False):
            """
            Creates a lazy iterator of combinations of entities in the taxonomy

            parameters:
                - total_generations          : 
                - threshold                  : 
                - beam_size                  : 
                - temperature pairing        : 
                - temperature_sample_size    : 
                - frequency_select           : 
                - upper_bound_skill_matching : 
            """
            
            all_skills = list(self.emb_tax.name.unique())
            
            
            
            
            for i in range(nb_generation):
                skill = np.random.choice(all_skills, size=1)[0]
                all_skills.remove(skill)
                if(len(all_skills) == 0):
                    ## continue the generation
                    all_skills = list(self.emb_tax.name.unique())
                if(not diverse):
                    yield [skill] + self.get_combination_for_(skill, 
                                                            threshold=threshold,
                                                            k=beam_size,
                                                            temperature=temperature_pairing,
                                                            frequency_select=frequency_select,
                                                            temperature_sample_size=temperature_sample_size,
                                                            upper_bound_skill_matching=upper_bound_skill_matching) ## get the tuple to generate

class AdvancedSkillsGenerator(SkillsGenerator):

    def __init__(self,
                 taxonomy:DataFrame,
                 taxonomy_is_embedded: bool,
                 combination_dist: npt.ArrayLike,
                 popularity: Dict,
                 emb_model: Any = None,
                 emb_tokenizer: Any = None,
                 reference_df: DataFrame = None,
                 entities_category_key: str=None):
        
        super().__init__(taxonomy, taxonomy_is_embedded, combination_dist, popularity, emb_model, emb_tokenizer, reference_df)

        self.entities_categories = list(self.emb_tax[entities_category_key].unique()) ## all the categories

        categories_embeddings = {}
        for cat in self.entities_categories:
            categories_embeddings[cat] = self.emb_tax[self.emb_tax[entities_category_key] == cat]
        self.categories_embeddings = categories_embeddings
    
        self.advanced_combinations_activated = True
        self.inter_categories_distribution = self.get_inter_categories_distribution(T=0.1)

        self.name_to_cat = self.emb_tax[["name", entities_category_key]].drop_duplicates("name").set_index("name").to_dict()[entities_category_key]
    
    def get_inter_categories_distribution(self, T: float):
        if(not self.advanced_combinations_activated):
            raise RuntimeError("Advanced combination are not activate. You need to input a entity category key.")
        inter_cat_dists = {}
        start = time.time()
        for i, cat1 in enumerate(self.entities_categories):
            cat1_embs = torch.cat(list(self.categories_embeddings[cat1]["embeddings"].values)).numpy()
            cat1_sims = []
            for j, cat2 in enumerate(self.entities_categories):
                
                ## if we already computed it
                if(cat2 in inter_cat_dists):
                    cat1_sims.append(inter_cat_dists[cat2][i]) ## inter_cat_dists[cat2][cat1]
                else:
                    ## if it's not already computed
                    cat2_embs = torch.cat(list(self.categories_embeddings[cat1]["embeddings"].values)).numpy()
                    cos_sims = cosine_similarity(cat1_embs, cat2_embs)
                    N, M = cos_sims.shape
                    cat1_sims.append(((np.tri(N, M, -1) * cos_sims).flatten()).mean())
            inter_cat_dists[cat1] = SkillsGenerator.softmax(np.array(cat1_sims), T)
        return inter_cat_dists
                


    def get_kNN_in_all_levels(self,
                              skillname: str,
                              category:str, 
                              k: int,
                              threshold: float=0.0):
        
        if(not self.advanced_combinations_activated):
            raise RuntimeError("Advanced combination are not activate. You need to input a entity category key.")
        skill_embedding = self.emb_tax[self.emb_tax["name"] == skillname]["embeddings"].values[0].numpy()
        kNN_in_levels = {}
        for i, category in enumerate(self.entities_categories):
            sims = cosine_similarity(skill_embedding, torch.cat(list(self.categories_embeddings[category]["embeddings"].values)).numpy())[0]
            if(self.entities_categories.index(category) == i):
                ## we deal directly with the skills's category, we have to exclude it
                idxs = sims.argsort()[::-1][1:k+1]
            else :
                ## other categories
                idxs = sims.argsort()[::-1][:k]

            ## threshold posteriori filtering
            idxs = [idx for idx in idxs if sims[idx] > threshold]
            
            kNN_in_levels[category] = [
                self.categories_embeddings[category].iloc[idx]['name']
                for idx in idxs
            ]
        return kNN_in_levels

    def get_combination_diverse(self,
                                skill: str,
                                category: str,
                                k:int,
                                frequency_select: bool,
                                temperature: float,
                                temperature_sample_size: float,
                                threshold:float,
                                upper_bound_skill_matching:int =None
                                ):
        if(not self.advanced_combinations_activated):
            raise RuntimeError("Advanced combination are not activate. You need to input a entity category key.")
        
        if(upper_bound_skill_matching is None):
            comb_size = self.get_combination_size(temperature_sample_size) - 1 ## we remove the skill selected first
        else :
            comb_size = min(self.get_combination_size(temperature_sample_size) - 1, upper_bound_skill_matching) ## we remove the skill selected first
        
        if(comb_size == 0):
            return []
        
        
        select_dist = self.inter_categories_distribution[category]
        kNNs = self.get_kNN_in_all_levels(skill, category, k, threshold)

        comb_levels = np.random.choice(self.entities_categories, size=comb_size, p=select_dist, replace=True)
        combination = []
        for comb_level in comb_levels:
            if(not frequency_select): ## deterministic
                select = kNNs[comb_level][0]
            else : ## popularity selection
                F = np.array([self.popularity[nn] for nn in kNNs[comb_level]])
                F = SkillsGenerator.softmax(-F, T=temperature) ## we get a dist, potentially dumped
                # with frequency dist
                select = np.random.choice(kNNs[comb_level], size=1, replace=False, p=F)[0]
            kNNs[comb_level].remove(select)
            combination.append(select)
        return combination  

    def balanced_nbred_iter(self, 
                            nb_generation=5000, 
                            threshold=0.0,
                            beam_size=20,
                            temperature_pairing=1,
                            temperature_sample_size=1,
                            frequency_select=True, 
                            upper_bound_skill_matching=None):
            """
            Creates a lazy iterator of combinations of entities in the taxonomy

            parameters:
                - total_generations          : 
                - threshold                  : 
                - beam_size                  : 
                - temperature pairing        : 
                - temperature_sample_size    : 
                - frequency_select           : 
                - upper_bound_skill_matching : 
            """
            
            all_skills = list(self.emb_tax.name.unique())
            print(all_skills)
            for i in range(nb_generation):
                skill = np.random.choice(all_skills, size=1)[0]
                
                all_skills.remove(skill)
                if(len(all_skills) == 0):
                    ## continue the generation
                    all_skills = list(self.emb_tax.name.unique())


                yield [skill] + self.get_combination_diverse(skill,
                                                            self.name_to_cat[skill],
                                                            k=beam_size,
                                                            threshold=threshold,
                                                            temperature=temperature_pairing,
                                                            frequency_select=frequency_select,
                                                            temperature_sample_size=temperature_sample_size,
                                                            upper_bound_skill_matching=upper_bound_skill_matching)
        

MODELS = {
    'gpt-3.5' : "gpt-3.5-turbo",
    'gpt-4'   : "gpt-4"
}

UPB_DENSE_GEN = 4
LB_SPARSE_GEN = 6 ## normally 3 but set to 6 to avoid them in skillspan

class DatasetGenerator():


    def __init__(self,
                 emb_tax: DataFrame,
                 reference_df: DataFrame = None,
                 emb_model: Any = None,
                 emb_tokenizer: Any = None,
                 additional_info:Dict[str, str]=defaultdict()):
        openai.api_key = API_KEY

        self.emb_tax = emb_tax

        ## reference dataset, use for precise few shots
        if(reference_df is not None):
            if("skill+sentence" not in reference_df.columns):
                raise ValueError("The taxonomy must contain a 'skill+sentence' column")
            self.references = SkillsGenerator.embedd_df(reference_df, "skill+sentence", emb_model, emb_tokenizer)
        
        ## additional info regarding the skills for finer prompt
        self.additional_infos = additional_info

        self.compute_sim_matrix()

        self.idx_to_label = {k: v for k, v in enumerate(emb_tax.name.values)}
        self.label_to_idx = {v: k for k, v in enumerate(emb_tax.name.values)}


    def generate_ds(self,
                    skill_generator,
                    specific_few_shots,
                    nb_few_shots=None,
                    shot_sim_threshold=0.0,
                    model="gpt-3",
                    gen_mode="baseline",
                    prompt_args={},
                    autosave=False,
                    autosave_file=None,
                    checkpoints_freq=100):
        ress = []
        for i, skills in tqdm(enumerate(skill_generator)):
            if(gen_mode == "PROTOTYPE"):
                if(len(skills) <= UPB_DENSE_GEN):
                    ## short skill list can use dense and sparse sentences
                    prompts = self.create_prompt_for(skills=skills,
                                                    mode="PROTO-GEN-A0", ## simple sentence generation for complete single skills
                                                    specific_few_shots=specific_few_shots,
                                                    number_few_shots=nb_few_shots,
                                                    shot_sim_threshold=shot_sim_threshold,
                                                    prompt_args=prompt_args)
                    
                    ress.append([skills, self.query(prompts, MODELS[model])])
                if(len(skills) >= LB_SPARSE_GEN): 
                    ## can't use dense sentences
                    prompts = self.create_prompt_for(skills=skills,
                                                    mode="PROTO-GEN-A1", ## simple sentence generation for complete single skills
                                                    specific_few_shots=specific_few_shots,
                                                    number_few_shots=nb_few_shots,
                                                    shot_sim_threshold=shot_sim_threshold,
                                                    prompt_args=prompt_args)
                    
                    ress.append([skills, self.query(prompts, MODELS[model])])
            else:
                ## can't use dense sentences
                prompts = self.create_prompt_for(skills=skills,
                                                mode=gen_mode, ## simple sentence generation for complete single skills
                                                specific_few_shots=specific_few_shots,
                                                number_few_shots=nb_few_shots,
                                                shot_sim_threshold=shot_sim_threshold,
                                                prompt_args=prompt_args)
                
                ress.append([skills, self.query(prompts, MODELS[model])])
            if(autosave and autosave_file is not None):
                if(i % checkpoints_freq == 0 and i != 0):
                    with open(autosave_file, "w") as f:
                        json.dump(ress, f)
                    print(f"> saved checkpoint at {i}")


        # ress += self.augment_with_no_label_negative_sample(n=500)
        return ress


    def augment_with_no_label_negative_sample(self, n=500, model="gpt-3"):
        """
            Query ChatGPT to generate samples that are linked to absolutely no samples 
            ex : 
                TYPE 1 : Company description
                1) "Here, you'll have the chance to build a career as unique as you are, with the global scale, 
                    support, inclusive culture and technology to become the best version of you"
                2) "With 1500 employees from more than 30 countries and cultures working at 14 sites, our company
                    provides air traffic control services for Switzerland and parts of neighbouring countries"
                3) "L'OCCITANE Group is a global, natural and organic ingredient-based cosmetics and well-being
                    products maker, producer and retailer"
                
                TYPE 2 : Salary and Perks
                1) "We offer flexible working hours based on a 40-hour week, vacation entitlement: 25 days from
                    the age of 20, 27 days from the age of 40 and 30 days from the age of 50."
        """
        
        nExamples = 20 ## number of generated samples in one generation
        samples = []
        for gtype, ratio in [["TYPE-1", 0.8], ["TYPE-2", 0.2]]:
            print(gtype)
            print(ceil(ratio * n / nExamples))
            for _ in range(ceil(ratio * n / nExamples)):
                ## type 1
                messages = [
                    {
                        'role': "system",
                        "content": PROMPT_TEMPLATE["NO-LABELS"][gtype]["system"].format(nExamples=nExamples)
                    },
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE["NO-LABELS"][gtype]["instruction"].format(nExamples=nExamples)
                    }
                ]
                shots = PROMPT_TEMPLATE["NO-LABELS"][gtype]["shots"]
            
                for shot in shots:
                    skills, posting,_ = shot.split("\n")
                    messages.append({'role':'user', 'content':skills})
                    messages.append({'role':'assistant', 'content':posting})

                messages.append({'role': 'user', 'content': "description: "})
                samples.append(self.query(messages, MODELS[model]))
        return samples
        

    def query(self, 
              messages:List[Dict[str, str]],
              model: str="gpt-4"):
        
        #######
        # print("-"*100)
        # for message in messages:
        #     print(message["content"])
        #######

        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20
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


    def create_prompt_for(self, 
                          mode: str,
                          skills: List[str],
                          specific_few_shots: bool,
                          number_few_shots: int,
                          shot_sim_threshold: float, 
                          prompt_args: Dict[str, str]):
        
        (system_prompt, instruction_field, shots_field) = (..., ..., ...)
        

        ## basic system prompt to get in the role
        system_prompt = self.prepare_prompt(PROMPT_TEMPLATE[mode]["role_instruction"], skills, prompt_args)
        ## only one sample for now
        instruction_field = self.prepare_prompt(PROMPT_TEMPLATE[mode]["instruction"], skills, prompt_args)

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
        shots = None
        if(specific_few_shots):
            shots = self.generate_specific_few_shots(skills, number_few_shots, shot_sim_threshold)
        else :
            shots = PROMPT_TEMPLATE[mode]["shots"]
        
        for shot in shots:
            skills, posting,_ = shot.split("\n")
            messages.append({'role':'user', 'content':skills})
            messages.append({'role':'assistant', 'content':posting})

        messages.append({'role': 'user', 'content': "skills: " +str(skills)})

        return messages


    def prepare_prompt(self, prompt_tf:str, skills: List[str], prompt_args: Dict[str, str]):
        argnames = re.findall('{(.+?)}', prompt_tf)
        # print("prompts arguments > ", argnames)
        args = {}

        for argname in argnames:
            if(argname in prompt_args):
                argval = prompt_args[argname]
            else :
                if(argname == "skillList"):
                    argval = str(skills)

                elif(argname == "typeOfAdditionalInfo"):
                    argval = "Alternative names (you may discard this information if irrelevant)"

                elif(argname == "nExamples"):
                    argval = "five" ## full leter in the paper
                    
                elif(argname == "implicitCount"):
                    argval = "two"
                    
                elif(argname == "additionalInfo"):
                    ## for alt names :
                    argval = ""
                    for i, skill in enumerate(skills):
                        argval += f"{i + 1}) {skill} : can also be referred as : {self.additional_infos[skill]['altLabels']} and described as : {self.additional_infos[skill]['description']}."

                elif(argname == "minNbSentences"):
                    ## for GEN-B1 PROTOTYPE
                    nb_sentence = str(max(1, len(skills) - 2))
                    argval = nb_sentence + (" sentence" if nb_sentence == "1" else " sentences")
                    
                elif(argname == "maxNbSentences"):
                    argval = str(len(skills) + 1)
                elif(argname == "wordsToAvoid"):
                    

                    ## computation of kNN of skills :
                    kNN_skills = []
                    # print(skills)
                    for skill in skills:
                        skill_idx = self.label_to_idx[skill]
                        sims_with_skill = self.pairwise_sims[skill_idx, :]
                        kNN = (-sims_with_skill).argsort()[1:2+1] ## 2NN
                        kNN_skills += [self.idx_to_label[nn] for nn in kNN if self.pairwise_sims[skill_idx, nn] if self.idx_to_label[nn] not in skills]
                    
                    if(len(kNN_skills) > 0):
                        argval = "You must not use any of these ESCO skills in the job description : "
                        argval += ", ".join(list(set(kNN_skills)))
                        argval += ". "
                    else :
                        argval = ""
                    
                else :
                    print("> Argument not found : {", argname, "}")
                    argval = ""
            args[argname] = argval
        return prompt_tf.format(**args)



    def generate_specific_few_shots(self, skills, n_shots, sim_treshold):
        skills_embs = torch.cat(list(self.emb_tax[self.emb_tax.name.isin(skills)]["embeddings"].values)).detach().numpy()
        all_refs = torch.cat(list(self.references["embeddings"].values)).detach().numpy()
        sims = cosine_similarity(skills_embs, all_refs).mean(axis=0) ## take the average with all sentences
        
        top_sims = ((-sims).argsort())[:n_shots]
        top_sims = np.array([nn for nn in top_sims if sims[nn] > sim_treshold])
        if(top_sims.shape[0] == 0):
            print("> no shots found the within the threshold")
            return []
        return list(self.references.iloc[top_sims]["skill+sentence"].apply(lambda x : "skills: " + str(x.split(" : ")[0]) +"\nJob Opening : " + str(x.split(" : ")[1])+".\n").values)

        
    def compute_sim_matrix(self):
        """
            Creates a new field in the class = the pairwise
            similarity matrix between all skill embeddings
        """
        skills_embeddings = torch.cat(list(self.emb_tax["embeddings"].values)).numpy()
        pairwise_sims = cosine_similarity(skills_embeddings, skills_embeddings)
        self.pairwise_sims = pairwise_sims

        

