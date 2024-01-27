from transformers import AutoTokenizer, AutoModel
import pickle
import os
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()


word_emb = "jjzha/jobbert-base-cased"
word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)
word_emb_model = AutoModel.from_pretrained(word_emb)

ESCO_DIR = "../../../esco/"


def embedd_tax(tax, show_progress=False, key_to_embedd="name+definition"):
    """
        sentences to embbed
    """
    if(show_progress):
        tax["embeddings"] = tax[key_to_embedd].progress_apply(
                    lambda st : 
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))
                    .last_hidden_state[:, 0, :].detach()
        )
    else :
        tax["embeddings"] = tax[key_to_embedd].apply(
                    lambda st : 
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))
                    .last_hidden_state[:, 0, :].detach()
        )
        
    return tax

def save_embedded_tax(emb_tax, fname):
    """
        Embedded taxonomy to save
    """

    with open(fname, "wb") as f:
        pickle.dump(emb_tax, f)
    

def embedd_large_tax(large_tax, final_name, chunk=400, verbose=False):
    
    fnames = []
    for n in tqdm(range(0, len(large_tax["name+definition"].index), chunk)):
        subtax = large_tax.iloc[n:n+chunk]
        embedded_subtax = embedd_tax(subtax, show_progress=False)
        fname = ESCO_DIR + f"temp/TEMP_LARGE_TAX_slice{n}.pkl"
        save_embedded_tax(embedded_subtax, fname)
        fnames.append(fname)
    
    if(n+chunk < len(large_tax.index)):
        subtax = large_tax.iloc[n+chunk:]
        embedded_subtax = embedd_tax(subtax)
        fname = ESCO_DIR + "temp/TEMP_LARGE_TAX_crumbs.pkl"
        save_embedded_tax(embedded_subtax, fname)
        fnames.append(fname)

    all_chunks = []
    for fname in fnames:
        with open(fname, "rb") as f:
            all_chunks.append(pickle.load(f))
    
    complete_emb_tax = pd.concat(all_chunks)

    if(len(complete_emb_tax.index) == len(large_tax.index)):
        print("large tax completly embedded") if verbose else None
        with open(ESCO_DIR + final_name, "wb") as f:
            pickle.dump(complete_emb_tax, f)
        print(f"Save final tax in {final_name}") if verbose else None
        [os.remove(fname) for fname in fnames]
        print("> Removed all temps files") if verbose else None
    else : 
        print(f"An error occured, files chunk were not merged, see slices in {ESCO_DIR}/temp/")
    



def load_skill_span(split):
    path = ESCO_DIR + split + ".json"
    with open(path) as f:
        split_records = eval(", ".join(f.read().split("\n")))
    return split_records    


