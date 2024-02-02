# SkillSkape

This is the source code for paper  [JOBSKAPE: A Framework for Generating Synthetic Job Postings to
Enhance Skill Matching](./_NLP4HR___Synthetic_job_posting_generation_for_skill_matching.pdf)


## Getting Started
Start with creating a **python 3.7** venv and installing **requirements.txt**.

## Directory


The directory is composed of multiple componenets :
- [JobSkape generator implementation](./generator.py)
- [Examples notebook for usage](./data_generation.ipynb)
- [SkillSkape dataset](./SkillSkape/)


```bash
.
├── SkillSkape ## SkillSkape Dataset
│   ├── SKILLSKAPE
│   ├── dev.csv
│   ├── test.csv
│   └── train.csv
├── taxonomy ## Used taxonomy
│   ├── dev_skillspan_emb.pkl
│   └── test_skillspan_emb.pkl
│
├── annotator ## Used to annotate the samples
│   ├── feedback.py
│   └── feedback_prompt_template.py
│
│
├── generator.py ## Implementation of Jobskape, dataset generation
├── gen_prompt_template.py ## prompt template for Jobskape and SkillSkape generation
│
│
├── models ## Implementation of both models
│   ├── skillExtract ## In-Context Learning Pipeline
│   │   ├── prompt_template.py
│   │   └── utils.py
│   └── supervised ## Supervised multiclassification model
│       ├── multilabel_classifier.py
│       ├── run_classifier.sh
│       └── run_inference.sh
│
├── annotation_refinement.ipynb ## Shows how to use the annotator
├── data_generation.ipynb ## Shows how to use JobSkape for data generation
├── dataset_evaluation.py ## Implementation of experiments and metrics
├── experiment.py ## Script to use the ICL pipeline with SkillSkape
├── static
│   └── ...
├── utils 
│   └── ppl_3_sentences_all_skills.csv ## Popularity distribution
│
│
├── README.md
└── requirements.txt 
```

## JobSkape Framework

![JobSkape framework](./static/jobskape.png)


The framework is made of two components, the Skill Generator that produces, for a given taxonomy, set of associated entities, and a Sentence generator that produces a sentence for each of these combinations.

### Skill Generator
We present a dataset generation framework to produce datasets for entity matching tasks. The mechanism is displayed in [this notebook](./data_generation.ipynb).

It works with two components. First we generate <ins>**faithful combinations of entities**</ins> :

```python
from generator import SkillsGenerator

gen = SkillsGenerator(taxonomy=emb_tax, ## input taxonomy embedded with a precise model 
            taxonomy_is_embedded=True, ## if the taxonomy is not yet embedded, provide a model
            combination_dist=combination_dist, ## combination size distribution
            emb_model=None, ## if not yet embedded, an encoder Mask Language Model
            emb_tokenizer=None, ## related tokenizer
            popularity=F ## popularity distribution
    )

gen_args = {
    "nb_generation" : split_size, # number of samples
    "threshold" : 0.83, # not considering skills that are less than .8 similar
    "beam_size" : 20, # considering 20 skills
    "temperature_pairing": 1, # popularity to be skewed toward popular skills
    "temperature_sample_size": 1,
    "frequency_select": True, # wether we select within the NN acording to frequency
}


generator = gen.balanced_nbred_iter(**gen_args) ## a generator
```


Then we generate <ins>**faithful sentence for each entity combination**</ins>:

```python

datagen = DatasetGenerator(emb_tax=emb_tax,
                           reference_df=None, ## no references, we work in Zero-Shot, you can input demponstration for kNN demonstration retrieval
                           emb_model=word_emb_model ## encoder Mask Language model to get embeddings,
                           emb_tokenizer=word_emb_tokenizer ## associated tokenizer,
                           additional_info=None ## potential additional information that can be added in custom promopt
            )

    generation_args = {
        "skill_generator": combs[:n_samples], 
        "specific_few_shots": False,
        "model": "gpt-3.5",
        "gen_mode": ["PROTO-GEN-A0" for Dense or "PROTO-GEN-A1" for Sparse],
        "autosave": True,
        "autosave_file": f"generated/SKILLSPAN/{split}.json",
        "checkpoints_freq":10
    }
    res = datagen.generate_ds(**generation_args)
```


Then you generate negative samples:



#### With excluded taxonomy:

```python
gen = SkillsGenerator(taxonomy=excluded_tax_sp, 
                taxonomy_is_embedded=True,
                combination_dist=combination_dist,
                popularity=F)
    
gen_args = {
        "nb_generation" : 500, # number of samples
        "threshold" : 0.83, # not considering skills that are less than .8 similar
        "beam_size" : 20, # considering 20 skills
        "temperature_pairing": 1, # popularity to be skewed toward popular skills
        "temperature_sample_size": 1,
        "frequency_select": True, # wether we select within the NN acording to frequency
    }


combinations = list(gen.balanced_nbred_iter(**gen_args))


datagen = DatasetGenerator(excluded_tax_sp,
                           None, ## no references, we work in Zero-Shot
                           word_emb_model,
                           word_emb_tokenizer,
                           additional_info)

generation_args = {
        "skill_generator": combinations, 
        "specific_few_shots": False,
        "model": "gpt-3.5",
        "gen_mode": ["PROTO-GEN-A0" for Dense or "PROTO-GEN-A1" for Sparse],
        "autosave": True,
        "autosave_file": f"generated/SKILLSPAN/excluded_tax_samples.json",
        "checkpoints_freq":10
    }
res = datagen.generate_ds(**generation_args)

```

#### With no labels :

```python


## put code


```



## SkillSkape Dataset



## Models 

We propose two models to evaluate our dataset. The baselines are :
- Annotated : [SkillSpan-M](https://github.com/jensjorisdecorte/Skill-Extraction-benchmark/tree/main) : Zhang, Mike, et al. "Skillspan: Hard and soft skill extraction from english job postings." (2022).
- Synthetic : [Decorte](https://huggingface.co/datasets/jensjorisdecorte/Synthetic-ESCO-skill-sentences) : Decorte, Jens-Joris, et al. Extreme Multi-Label Skill Extraction Training using Large Language Models (2023)
### Supervised :

```
supervised
├── multilabel_classifier.py
├── run_classifier.sh
└── run_inference.sh
```

- `multilabel_classifier.py` : implementation of supervised model based on BERT for multiclass classification
- `run_classifier.sh` : training script
- `run_inference.sh` : inference script


### Unspervised - In Context Learning

![Alt text](static/icl_pipeline.png)

Uses an annotated dataset for training set via *kNN demonstration retrieval*.

Usage :


```bash
python experiment.py --metric_fname metric_file.txt --support SKILLSKAPE --bootstrap 5
```

with arguments :

- `metric_fname` : Name of the target metric file (required)
- `support` : support set, by default `SKILLSPAN`
- `bootstrap` : Number of bootstrap iterations for the experiment




