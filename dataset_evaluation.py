import sys
sys.path.append("../skillExtract/")
import pandas as pd
from transformers import (AutoModel, AutoTokenizer)
from utils import select_candidates_from_taxonomy
import pickle
from tqdm import tqdm
from utils import (OPENAI,
                   Splitter,
                   select_candidates_from_taxonomy)
from api_key import API_KEY
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (precision_score, 
                             recall_score)
import numpy as np
import torch
import random
from split_words import Splitter
from gen_utils import embedd_tax

# ESCO_DIR = "../../../esco/"
ESCO_DIR = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/magron/esco/"
GENERATED_DIR = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/magron/SkillThrills/protosp01/dataset_generation/generation/generated/"



OPTION_LETTERS = list("abcdefghijklmnopqrstuvwxyz".upper())

word_emb = "jjzha/jobbert-base-cased"
word_emb_model = AutoModel.from_pretrained(word_emb)
word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)




def fidelity(dataset, model_id='jjzha/jobbert-base-cased'):
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model_id,
                                add_start_token=True,
                                predictions=dataset)
    return results


def diversity(dataset):
    ## compute the embeddings of the sentences
    if("embeddings" not in dataset.columns):
        dataset["embeddings"] = dataset["sentence"]\
                    .progress_apply(lambda st : \
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))\
                    .last_hidden_state[:, 0, :]\
                    .detach()
                    )
    all_embeddings = torch.cat(list(dataset["embeddings"])).numpy()
    all_extra = cosine_similarity(all_embeddings, all_embeddings)
    N = len(dataset)
    div = ((np.ones((N, N)) - np.tri(N, N)) * all_extra).flatten()
    return div[div != 0]


def intra_skills_diversity(dataset, label_key="skills"):
    dataset["embeddings"] = dataset["sentence"]\
                    .progress_apply(lambda st : \
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))\
                    .last_hidden_state[:, 0, :]\
                    .detach()
                    )
    skill_embds = dataset\
        .explode(label_key)\
        .groupby(label_key)\
        .progress_apply(lambda x : torch.cat(list(x["embeddings"])))
    
    intra_sim = skill_embds\
                    .progress_apply(
                        lambda x : cosine_similarity(x, x)
                    )\
                    .apply(
                        lambda x : ((np.ones((x.shape[0], x.shape[0])) - np.tri(x.shape[0], x.shape[0])) * x).flatten().mean()                        
                    )

    intra_sim = pd.DataFrame(
        intra_sim
    ).reset_index()

    intra_sim.columns = ["skill", "intra_sim"]

    return intra_sim
    





def sentence_level_quality(dataset, label_key="label"):
    """
        dataset is given as a dataframe
    """
    def compute_cos_sim(entry):
        sentence_embedding = entry["embeddings"].detach().numpy()
        label = entry[label_key]
        if(label in known_label_set):
            label_embedding = emb_tax[emb_tax["name"] == label]["embeddings"].values[0].detach().numpy()
            return cosine_similarity(sentence_embedding, label_embedding)[0][0]

    
    emb_tax = pd.concat([get_sp_emb_tax("dev")[0], get_sp_emb_tax("test")[0]])
    known_label_set = set(emb_tax["name"].values)    
    ## compute the embeddings of the sentences
    dataset["embeddings"] = dataset["sentence"]\
                    .progress_apply(lambda st : \
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))\
                    .last_hidden_state[:, 0, :]\
                    .detach()
                    )
    dataset["sim"] = dataset[["embeddings", label_key]].dropna().progress_apply(compute_cos_sim, axis=1)

    return dataset


def entity_level_quality_1_to_1(predictions, label_key):
    preds_gt = []
    tbp = []
    for tpred_item in tqdm(predictions):
        pred_item = tpred_item[0]
        tbp.append([pred_item[label_key], list(pred_item["matched_skills"].keys())])
        labels = eval(pred_item[label_key]) if type(pred_item[label_key]) == "str" else pred_item[label_key]
        for label in labels:
            if(label not in ["LABEL NOT PRESENT", "UNDERSPECIFIED"]):
                
                # ONE ENTRY PER LABEL
                predicted_labels = [
                    x["name+definition"].split(" : ")[0] for x in pred_item["matched_skills"].values()
                ]
                if(label in predicted_labels):
                    # print("-"*45)
                    # print("we have a match for :", label)
                    # print("in : ", predicted_labels)

                    preds_gt.append([label, label])
                else :
                    if(len(predicted_labels) > 0):
                        preds_gt.append([label, predicted_labels[0]])
                    else :
                        preds_gt.append([label, "LABEL NOT PRESENT"])
                
                # THIS VERSION HAS ONE PREDICTION ENTRY PER 
                # (LABEL, PREDICTION) AND THUS HAS WAY TOO 
                # MUCH ENTRY TO PREVIDE MEANINGFUL RESULTS
                # for matched_pred in predicted_labels:
                #     preds_gt.append([label, matched_pred])
    gts, preds = zip(*preds_gt)
    gts = np.array(gts)
    preds = np.array(preds)

    acc = np.sum((gts == preds).astype(int)) / gts.shape[0]

    print(tbp)

    precision_micro = precision_score(gts, preds, average="micro")
    precision_macro = precision_score(gts, preds, average="macro")
    recall_micro = recall_score(gts, preds, average='micro')
    recall_macro = recall_score(gts, preds, average="macro")
    return {
        "accuracy": acc,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "F1_micro": (2*precision_micro*recall_micro / (precision_micro + recall_micro)),
        "F1_macro": (2*precision_macro*recall_macro / (precision_macro + recall_macro))
    }

def entity_level_quality_many_to_many(predictions, label_key):
    ## simply an accuracy based on Jaccard Coefficient
    jaccards = []
    for pred_item in predictions:
        labels = set(pred_item[0][label_key])
        preds = set(pred_item[0]["matched_skills"])
        jaccards.append(len(labels.intersection(preds)) / len(labels.union(preds)))
    
    return {
        'jaccard_accuracy':sum(jaccards) / len(jaccards)
    }

def entity_level_quality(predictions, label_key="label"):
    print("new")
    return {
        '1_to_1': entity_level_quality_1_to_1(predictions, label_key), 
        'many_to_many': entity_level_quality_many_to_many(predictions, label_key)
    }





class Args():
    def __init__(self,
                 taxonomy="",
                 datapath="remote_annotated-en",
                 candidates_method="mixed",
                 shots=6,
                 prompt_type="skills",
                 data_type = "en_job",
                 max_tokens = 512,
                 api_key = API_KEY,
                 model = "gpt-3.5-turbo",
                 temperature = 0,
                 top_p = 1,
                 frequency_penalty = 0,
                 presence_penalty = 0):
        self.taxonomy = taxonomy

        ## RELATED TO PROMPT CREATION
        self.datapath = datapath
        # self.candidates_method = "embeddings"  ## putting "rules" doesn't give the embeddings
        self.candidates_method = candidates_method
        self.shots = shots
        self.prompt_type = prompt_type
        self.data_type = data_type
        
        ## RELATED TO CHAT GPT GENERATION
        self.max_tokens =max_tokens      ## ?? default value but JobBERT suposedly takes 512
        self.api_key = api_key
        self.model = model
        self.temperature =temperature     ## default val
        self.top_p =top_p           ## default val
        self.frequency_penalty = frequency_penalty# default val
        self.presence_penalty = presence_penalty## default val


def get_proto_emb_tax():
    with open(ESCO_DIR + "embedded_tech_management_tax.pkl", "rb") as emb:
        tech_emb_tax = pickle.load(emb)

    taxonomy = pd.read_csv(ESCO_DIR + "tech_managment_taxonomy_narrow.csv")
    taxonomy["name+definition"] = taxonomy["name+defintion"]
    taxonomy["Example"] = taxonomy["altLabels"]
    return tech_emb_tax, taxonomy


def get_sp_emb_tax(split):
    if(split == "dev"):
        with open(ESCO_DIR + "dev_skillspan_emb.pkl", "rb") as emb:
            sp_emb_tax = pickle.load(emb)
    if(split == "test"):
        with open(ESCO_DIR + "test_skillspan_emb.pkl", "rb") as emb:
            sp_emb_tax = pickle.load(emb)

    sp_emb_tax["Example"] = sp_emb_tax["altLabels"]
    taxonomy = sp_emb_tax.drop("embeddings", axis=1)
    return sp_emb_tax, taxonomy


class Predictor():

    def __init__(self,
                 test_domain="SkillSpan-dev",
                 train_domain="Proto",
                 candidates_method="mixed",
                 datapath="remote_annotated-en",
                 shots=6,
                 prompt_type="skills",
                 data_type = "en_job",
                 max_tokens = 512,
                 api_key = API_KEY,
                 model = "gpt-3.5-turbo",
                 temperature = 0,
                 top_p = 1,
                 frequency_penalty = 0,
                 presence_penalty = 0
                 ) -> None:
        
        self.args = Args(datapath=datapath,
                    candidates_method=candidates_method,
                    shots=shots,
                    prompt_type=prompt_type,
                    data_type=data_type,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty)
        
        if(test_domain == "SkillSpan-dev"):
            self.emb_tax, self.taxonomy = get_sp_emb_tax("dev")
        elif(test_domain == "SkillSpan-test"):
            self.emb_tax, self.taxonomy = get_sp_emb_tax("test")        
        elif(test_domain == "SkillSpan-test+dev" or test_domain == "SkillSpan-dev+test"):
            emb_tax1, taxonomy1 = get_sp_emb_tax("test")
            emb_tax2, taxonomy2 = get_sp_emb_tax("dev")
            self.emb_tax = pd.concat([emb_tax1, emb_tax2]).drop_duplicates("name")
            self.taxonomy = pd.concat([taxonomy1, taxonomy2]).drop_duplicates("name")
        elif(test_domain == "Proto"):
            self.emb_tax, self.taxonomy = get_proto_emb_tax()
        else:
            raise ValueError("Unknown or unsupported test domain.")

        ## ADD TRAIN DOMAIN TO DECIDE WHERE THE SUPPORT COME FROM
        support_set_fname_base = GENERATED_DIR + "{DOMAIN}/ANNOTATED_SUPPORT_SET.pkl"
        target = {
            "Proto": "PROTOTYPE",
            "SkillSpan": "SKILLSPAN",
            "R-SkillSpan": "R-SKILLSPAN",
            "Decorte": "DECORTE"
        }
        if(train_domain in target):
            support_set_fname = support_set_fname_base.format(DOMAIN=target[train_domain])
        else:
            raise ValueError("Unknown or unsupported train domain")
        

        with open(support_set_fname, "rb") as f:
            self.support_set = pickle.load(f).dropna()
            
            if(train_domain not in ["Decorte", "R-SkillSpan"]): ## 1st gen
                self.support_set["skill"] = self.support_set["skill"].apply(eval) ## into list
            else: ## last generation
                self.support_set = self.support_set.rename({'annot_spans': 'spans'}, axis=1)

            if(train_domain not in ["Decorte"]):
                self.support_set = self.support_set.explode("skill")
            
            if(train_domain in ["Decorte"]):
                self.support_set["spans"] = self.support_set["spans"].apply(eval) ## into list
                

        self.unique_support = self.support_set[["sentence", "embeddings"]].drop_duplicates()
        self.all_embeddings = torch.cat(list(self.unique_support["embeddings"].values)).detach()

        


    def pipeline_prediction(self, dataset, support_type=None, support_size_match=None, nb_candidates=None, support_size_extr=None):
        """
            takes dataset as record list
        """

        if(support_type not in ["kNN", "rand", None]):
            raise ValueError("Unknown or unsupported support type.")
        args = self.args
        ## loading embeddings taxonomy less
        
        extraction_cost = 0
        matching_cost = 0
        ress = []
        for i, annotated_record in tqdm(list(enumerate(dataset))):
            ## EXTRACTION
            if(support_size_extr is not None):
                api = OPENAI(args, [annotated_record], self.generate_support_set(annotated_record["sentence"], type="extraction", k=support_size_extr))
            else:
                api = OPENAI(args, [annotated_record])
            sentences_res_list, cost = api.do_prediction("extraction")
            extraction_cost += cost
            if(nb_candidates is None):
                nb_candidates = 5
            ## CANDIDATE SELECTION
            if "extracted_skills" in sentences_res_list[0]:
                splitter = Splitter()
                for idxx, sample in enumerate(sentences_res_list):
                    sample = select_candidates_from_taxonomy(
                        sample,
                        self.taxonomy,
                        splitter,
                        word_emb_model,
                        word_emb_tokenizer,
                        max_candidates=nb_candidates,
                        method=args.candidates_method,
                        emb_tax=None if args.candidates_method == "rules" else self.emb_tax,
                    )

                    sentences_res_list[idxx] = sample


            if("skill_candidates" in sentences_res_list[0]):

                if(support_type is not None and support_type == "random"):
                    shots_fields = self.generate_random_support_set(sentences_res_list[0]["sentence"], support_size_match, nb_candidates)  
                elif(support_type is not None and support_type == "kNN"):
                    shots_fields = self.generate_support_set(sentences_res_list[0]["sentence"], support_size_match, nb_candidates)
                else:
                    shots_fields = None
                api = OPENAI(args, sentences_res_list, shots_fields)
                sentences_res_list, cost = api.do_prediction("matching")
                matching_cost += cost



            ress.append(sentences_res_list)
        return ress

    def generate_support_set(self, sentence, k, nb_candidates=10, type="matching"):
        embedding = word_emb_model(
            **word_emb_tokenizer(sentence, return_tensors="pt", max_length=512, padding=True, truncation=True)
        ).last_hidden_state[:, 0, :] ## BOTTLENECK 1
        sims = torch.nn.functional.cosine_similarity(embedding, self.all_embeddings)
        sims = sims.argsort().flip(dims=(-1,))[:k]
        supports = [
            self.support_set[["sentence", "annotated_sentence", "spans", "skill"]][self.support_set["sentence"] == x].sample(1).to_dict("records")[0]
            for x in list(self.unique_support.iloc[sims]["sentence"])
            ]
        

        prepared_supports = []
        for support in supports:
            if(len(support["spans"]) > 0):
                prepared_supports.append(
                    self.prepare_support(support, nb_candidates) if type == "matching" else self.prepare_support_extract(support)
                )
        return prepared_supports


    def generate_random_support_set(self, sentence, k):
        return [
            self.prepare_support(x) 
            for x in self.support_set.sample(k)[["sentence", "annotated_sentence", "spans", "skill"]].to_dict("records")
            if len(x["spans"]) > 0
        ]
    
    def prepare_support(self, support, nb_candidates):

        splitter = Splitter()
        span = random.choice(support["spans"])
        candidates = select_candidates_from_taxonomy( ## BOTTLENECK 2
            {
                "extracted_skills":[span],
                "sentence": support["sentence"]
            }, 
            self.taxonomy,
            splitter=splitter,
            tokenizer=word_emb_tokenizer,
            model=word_emb_model,
            max_candidates=nb_candidates,
            method="embeddings", ### COMPARE WITH MIXED AND CANDIDATEDS WILL THEN BE 20 AS FOR THE RUN 
            emb_tax = self.emb_tax
        )
        if(support["skill"] in self.taxonomy.name):
            skill_nd = self.taxonomy[self.taxonomy.name == support["skill"]].iloc[0]["name+definition"]
        else :
            skill_nd = support["skill"]
        all_candidates = [candidate["name+definition"] for candidate in candidates["skill_candidates"][span]]
        
        ## if the current skill is not the selected candidates : ADD IT
        if(skill_nd not in all_candidates):
            all_candidates = [skill_nd] + all_candidates[:-1]
        
        shot = f"Sentence: {support['sentence']}\n"
        shot += f"Skill : {span}\nOptions: \n"
        options = {}
        for l, cand in zip(OPTION_LETTERS, all_candidates):
            shot += l + ": " + cand + "\n"
            options[cand] = l
        shot += f"Answer: {options[skill_nd]}\n" ## answer with the letter instead of copying
        return shot

    def prepare_support_extract(self, support):
        return "Sentence: " + support["sentence"] + "\nAnswer: " + support["annotated_sentence"]

