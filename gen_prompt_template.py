PROMPT_TEMPLATE = {
    "baseline": {
        "role_instruction": "You are a hiring manager for a big company. You need to define write a job opening for different skill requirements in your company.\n",
        "instruction": "You are a hiring manager for a big company and your goal is to write the perfect sentence to describe job that uses a set of skills. You'll be given a set of skill, the job posting will reference each of them explicitely or implicitely. The job you describe must include capailities in each of these skills. No external skills should be mentionned. The description of the job should be one line long and be as specific as possible.\n",
        "shots": [
            "skills: [SLQ, relational databases]\nJob opening : ability to manage database and query using SQL.\n"
        ]
    },
    "PAPER-GEN" : { ## PAPER-GEN : zero shot following arxviv:2037.03539
        "role_instruction" : "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} examples that could be found in a job ad and refer to the skill or knowledge component. You may be given a skill family to help you disambiguate if the skill name could refer to multiple things. Ensure that your examples are well written and could be found in real job advertisement. Write a variety of different sentences and ensure your examples are well diversified. Use a variety of styles. Write examples using both shorter and longer sentences, as well as examples using short paragraphs of a few sentences, where sometimes only one is directly relevant to the skill. You're trying to provide a representative sample of the many, many ways real job postings would evoke a skill. At least {implicitCount} of your examples must not contain an explicit reference to the skill and must thus not contain the given skill string. {typeOfAdditionalInfo}: {additionalInfo} Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [

        ]
    },
    ############################################################################################################################
    "GEN-A0": { ## [V] GEN-A0 : Sentence
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to list {nExamples} examples of exactly one sentence that could be found in a job ad and refer to all skill or knowledge component. You may be given a skill family to help you disambiguate if the skills names could refer to multiple things. Ensure that your sentences are well written and could be found in real job advertisement. Write a variety of different examples and ensure your sentences are well diversified. Use a variety of styles. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each sentences. A candidate should have different degrees of expertise in all the given skills. This degree should be specified for each skills in each sentence. You must not include any skills in ESCO that were not given to you. Try to be as implicit as possible when mentionning the skill. Try not to use the exact skill string. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    "GEN-A1": { ## [V] GEN-A1 : Paragraph 
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} paragraphs of a few lines that could be found in a job ad and refer to all skill or knowledge component. You may be given a skill family to help you disambiguate if the skills names could refer to multiple things. Ensure that your paragraphs are well written and could be found in real job advertisement. Write a variety of different paragprahs and ensure your examples are well diversified. Use a variety of styles. Write paragraphs of a few sentences. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each paragraph. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. At least {implicitCount} of your paragraphs must not contain an explicit reference to the skill and must thus not contain the given skill string. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo} Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    ############################################################################################################################
    "GEN-A2": { ## [X] GEN-A2 : Full Job Posting
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to list {nExamples} complete job ads and refer to all skills or knowledge components. You may be given additional skill information such as alternative names and descriptions to help you disambiguate if the skills names could refer to multiple things. Ensure that your job postings are well written and could be a real job advertisement. Write a variety of different job advertisment and ensure your examples are well diversified. Use a variety of styles. Write job openings of a few paragraphs. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into all the job openings. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. At least {implicitCount} of your job postings must not contain an explicit reference to the skill and must thus not contain the given skill string. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    "GEN-A3": { ## [X] GEN-A3 : Full Job Posting
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} job descriptions and refer to all skills or knowledge components. You may be given additional skill information such as alternative names and descriptions to help you disambiguate if the skills names could refer to multiple things. Ensure that your job description are well written and could be part of a real job advertisement. Write a variety of different job advertisment and ensure your examples are well diversified. Use a variety of styles. It is very importnt that the job description contain multiple paragraphs of a few sentence each. You must absolutely not do lists, write sentences. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into all the job description. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the ad. At least {implicitCount} of your job postings must not contain an explicit reference to the skill and must thus not contain the given skill string. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    ##############################################################################################################################
    "GEN-B1": { ## [~] GEN-B1 : PROTOTYPE, for skill combination of 1, 2 and 3 skills we ask for variations of small sentences
    "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
    "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} paragraphs that could be found in a job ad and refer to all skill or knowledge component. The paragraphs must have varying length between {minNbSentences} and {maxNbSentences} sentences. The first description must be exactly and precisely {minNbSentences} long. The last description must be exactly and precisely {maxNbSentences} sentences long. All the description should mention at least once each and every skills. Ensure that your paragraphs are well written and could be found in real job advertisement. Write a variety of different paragprahs and ensure your examples are well diversified. Use a variety of styles. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each paragraph. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. The skills should not be mentionned explicitely. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo} Avoid explicitly using the wording of this extra information in your examples.\n",
    "shots": [

        ]
    },
    ##############################################################################################################################
    "GEN-C0": { ## [X] GEN-C1 :  Advanced A0, extracting the exact span 
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to list {nExamples} examples of exactly one sentence that could be found in a job ad and refer to all skill or knowledge component. You may be given a skill family to help you disambiguate if the skills names could refer to multiple things. Ensure that your sentences are well written and could be found in real job advertisement. Write a variety of different examples and ensure your sentences are well diversified. Use a variety of styles. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each sentences. A candidate should have different degrees of expertise in all the given skills. This degree should be specified for each skills in each sentence. You must not include any skills in ESCO that were not given to you. Try to be as implicit as possible when mentionning the skill. Try not to use the exact skill string. {wordsToAvoid}{typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job you just described, by surrounding them with tags '@@' at the beginning of the tag and '##' at the end of the tag. After the '##' you must write between [ ] the skill that's related to this tag. The skill must be one given in input. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
        "shots": [
            
        ]
    },
    "PROTO-GEN-A0":{ ## DENSE
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to produce a single example of exactly one sentence that could be found in a job ad and refer to all skill or knowledge component. Ensure that your sentence is well written and could be found in real job advertisement. Use a variety of styles. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills in : {skillList} must be integrated. A candidate should have different degrees of expertise in all the given skills. This degree should be specified for each skills in the sentence. You must not include any skills in ESCO that were not given to you. Try to be as implicit as possible when mentionning the skill. Try not to use the exact skill string. {wordsToAvoid}. Avoid explicitly using the wording of this extra information in your examples. Your sentence must not start with 'We are seeking', 'We are looking' or 'We are searching'. Generate stricly only one example.\n",
        "shots": [
            
        ]
    },
    "PROTO-GEN-A1":{ ## SPARSE
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide a single paragraph of a few lines that could be found in a job ad and refer to all skill or knowledge component. You may be given a skill family to help you disambiguate if the skills names could refer to multiple things. Ensure that your paragraphs are well written and could be found in real job advertisement. Use a variety of styles. Write paragraphs of a few sentences. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each paragraph. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. The references to the skills must be as implicit as possible and must thus not contain the given skill string. {wordsToAvoid} Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },

    "NO-LABELS":{ ## NEGATIVE SAMPLES
        "TYPE-1": { ## COMPANY DESCRIPTIONS
            "system": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at writing introductions of job posting",
            "instruction":"You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at writing introductions of job posting. You should write {nExamples} examples of the first line of the job posting. It should consists in introducing the company, its localization, the number of employees, and any information relevant to a future candidates who wants to learn about the company. The description should be concise, specify the potential growth of the company and a domain of action. You shouldn't mentoin anything about the actual job, no skills required for the candidate and shouldn't mention the candidate at all. You should mention a wide range of company field, size, and localization in each of the examples.",
            "shots": [
                
            ]
        },
        "TYPE-2": { ## SALARY AND PERKS
            "system": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at specifying administrative information in job posting.",
            "instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at specifying administrative information in job posting. You should produce {nExamples} descriptions of the salary and the perks a candidate to a certain job would have. You shouldn't mention the actual job and the candidate itself. You could add diversity by varying the salary and the perks. You must write a salary range between 40k and 100k according to the job in half of your generation.",
            "shots": [
                
            ]

        }
    },


    ####################################################################################
    "PAPERV2":{
        "system": "Respond with sentences from hypothetical job ads that require a certain skill, as asked by the user.",
        "Instruction": "Respond with sentences from hypothetical job ads that require a certain skill, as asked by the user.",
        "shots":[
            
        ]

    }


    ####################################################################################


}



