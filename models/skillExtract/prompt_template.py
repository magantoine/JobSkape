job_shots_extr_skills = [
    "Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: Wir suchen einen Teamleiter mit ausgeprägten @@Kommunikationskompetenzen##, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.",
    "Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: Die @@Fähigkeit zur interdisziplinären Zusammenarbeit## ist ein Schlüsselkriterium für diese Position.",
    "Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \n Answer: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams.",
    "Sentence: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren. \nAnswer: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, @@richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren##.",
    "Sentence: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter. \nAnswer: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter.",
    "Sentence: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld kontinuierlich weiterzuentwickeln. \nAnswer: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld @@kontinuierlich weiterzuentwickeln##.\n",
]

job_shots_extr_wlevels = [
    'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: {"Kommunikationskompetenzen": "advanced"}',
    'Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: {"Fähigkeit zur interdisziplinären Zusammenarbeit": "unknown"}',
    'Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \nAnswer: {"Java Senior Software Engineer": "advanced"}',
    "Sentence: Du arbeitst eng mit unserem erfahrenen Team zusammen und trägst aktiv zum Erfolg des Unternehmens bei. \nAnswer: {}",
    'Sentence: Du hast sehr gute Kenntnisse in digitaler Schaltungstechnik und Regelkreisen. \nAnswer: {"digitaler Schaltungstechnik": "advanced", "Regelkreisen": "advanced"}',
    'Sentence: Nebst guten Kenntnisse in moderner, agiler Softwareentwicklung und deren Konzepte, hast du auch noch ein grundlegendes Wissen in der Testautomatisierung. \nAnswer: {"agiler Softwareentwicklung": "advanced", "Testautomatisierung": "beginner"}',
]

# send example to Marco to check on these
# ask for matching examples

job_shots_match = [
    'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
]

## SKILL EXTRACTION IN JOB OPENING PROMPTS IN ENGLISH
en_job_shots_extr_skills = [
    "Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team.\nAnswer: We are looking for a team leader with strong @@communication skills## to foster collaboration and information sharing within the team.",
    "Sentence: the ability to work collaboratively across disciplines is a key criterion for this position. \nAnswer: @@ability to collaborate across disciplines## is a key criterion for this position.",
    "Sentence: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team. \nAnswer: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team.",
    "Sentence: In her role as a team leader, she has continuously supported the professional development of her employees. \nAnswer: In her role as a team leader, she has continuously fostered the professional @@development of her employees##.",
    "Sentence: He is a resilient employee who has been able to set proper priorities and organize tasks thoughtfully during periods of heavy workload. \nAnswer: He is a resilient employee who has been able to set @@correct priorities and organize tasks thoughtfully## during periods of high workload.",
    "Sentence: Highly qualified, flexible employees from the insurance and IT industry develop them further. \nAnswer: Highly qualified, flexible employees from the insurance and IT industries continue to develop them.",
    "Sentence: Over the past few years, it has succeeded in continuously developing itself in a rapidly changing environment. \nAnswer: Over the past few years, he has succeeded in @@continuously developing## himself in a rapidly changing environment##.\n",
    ]

en_job_shots_match = [
    'Sentence: Understand basic provisions of copyright and privacy. \nSkill: Data protection. \nOptions: \nA: "Respect privacy principles." \nB: "Understand data protection" \nC: "Ensure data protection in aviation operations" \nD: "Data protection." \nAnswer: b, d.\n',
    ]

course_shots_extr_skills = job_shots_extr_skills
course_shots_extr_wlevels = job_shots_extr_wlevels
course_shots_match = job_shots_match

cv_shots_extr_skills = job_shots_extr_skills
cv_shots_extr_wlevels = job_shots_extr_wlevels
cv_shots_match = job_shots_match

PROMPT_TEMPLATES = {
    "en_job": {
        "system": "You are an expert human resource manager. You need to analyse skills in a job posting.",
        "extraction" : {
            "skills":{
                "instruction": "You are an expert human resource manager. You are given an extract from a job description. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself. Make sure to rewrite the sentence with all the tags.\n",
                "shots": en_job_shots_extr_skills,
            },
        },
        "matching" : {
            "instruction" : "You are an expert human resource manager. You are given a sentence from a job description, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": en_job_shots_match,
        }
    },
    "job": {
        "system": "You are an expert human resource manager. You need to analyse skills in a job posting.",
        "extraction": {
            "skills": {
                "instruction": "You are an expert human resource manager. You are given an extract from a job description in German. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
                # "instruction": "You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n"
                "shots": job_shots_extr_skills,
            },
            "wlevels": {
                "instruction": "You are given a sentence from a job description in German. Extract all skills, competencies, and tasks that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
                "shots": job_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. You are given a sentence from a job description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": job_shots_match,
        },
    },
    "course": {
        "system": "You are looking for an online course.",
        "extraction": {
            "skills": {
                "instruction": "You are given a sentence from a course description in German. Highlight all the skills and competencies that are learned when following the course described in the sentence, by surrounding them with tags '@@' and '##'.\n",
                "shots": course_shots_extr_skills,
            },
            "wlevels": {
                "instruction": "You are given a sentence from a job description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
                "shots": course_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are looking for an online course. You are given a sentence from a course description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": course_shots_match,
        },
    },
    "cv": {
        "system": "You are an expert human resource manager. You need to analyse skills in a CV.",
        "extraction": {
            "skills": {
                "instruction": "Extract candidates skills in German from the following German sentence, taken from a CV.\n",
                "shots": cv_shots_extr_skills,
            },
            "wlevels": {
                "instruction": "Extract all skills and competencies from the CV (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
                "shots": cv_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. From a given German sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
            "shots": cv_shots_match,
        },
    },
}
