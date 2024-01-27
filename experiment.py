from dataset_evaluation import Predictor
import pandas as pd
import pickle
import sys


AD = False

def main():

    ## MESSY ARGUMENTS HANDLING
    NB_ARGUMENTS = 3
    if(len(sys.argv) < NB_ARGUMENTS + 1):
        raise ValueError("Wrong set of arguments")
    target_metric_filename = sys.argv[1]
    SUPPORT = sys.argv[2]
    BOOTSTRAP = int(sys.argv[3])
    
    OUTSET = sys.argv[4] if len(sys.argv) > 4 else None 
    fname_preds = sys.argv[5] if len(sys.argv) > 5 else None 
    ##########################

    print(f"METRIC FILE NAME : {target_metric_filename}.pkl")
    print(f"BOOTSTRAPING REPETITIONS : {BOOTSTRAP}")
    if(fname_preds is not None):
        print(f"SAVE PRED FILE IN : {fname_preds}")
    ## we test with SkillSpan dataset
    if AD:
        ESCO_DIR = "../../data/taxonomy/esco/"
        with open(ESCO_DIR + "dev.json") as f:
            testsp = eval(",".join(f.read().split("\n")))
    else:
        ESCO_DIR = "../../../esco/"
        testsp = pd.read_csv("./generation/generated/SKILLSPAN/test_refined.csv", names=["idx", "sentence", "skills"], header=1)
    testsp = pd.DataFrame(testsp).drop("idx", axis=1).dropna()
    testsp.columns = ["sentence", "skills"]
    testsp = testsp[["skills", "sentence"]]

    testsp = testsp.to_dict(orient='records') ## input to predictor are as records

    print("Number of dev sp samples : ", len(testsp))
    predictor = Predictor(
            test_domain="SkillSpan-test+dev", ## goal is to use the subset of train samples 
            train_domain=SUPPORT, ## SkillSpan means SkillSpans SP 
            candidates_method="mixed"
        )
    
    all_res = []

    
    all_kwargs = [
        {
            "dataset": testsp,
            "support_type": "kNN",
            "nb_candidates": 10,
            "support_size_match": 1,
            "support_size_extr": 7,
        }
    ]

    for kwargs in all_kwargs:
        all_res.append(bootstrap(predictor, kwargs, BOOTSTRAP, fname_preds))

    
    with open(f"{target_metric_filename}.pkl", "wb") as f:
        pickle.dump(all_res, f)



def bootstrap(predictor, pred_kwargs, BOOTSTRAP, fname_preds=None):
    spec_res = []
    for _ in range(BOOTSTRAP):
        res1 = predictor.pipeline_prediction(**pred_kwargs) ## BASELINE
        spec_res.append(compute_metrics(res1))
    fname = prepare_kwargs(pred_kwargs, content="res")
    with open(fname, "wb") as f:
        pickle.dump(spec_res, f)
    fname = prepare_kwargs(pred_kwargs, content="preds") if fname_preds is None else fname_preds
    with open(fname, "wb") as f:
        pickle.dump(res1, f)
    return spec_res

    
def prepare_kwargs(pred_kwargs, content):
    fname = content
    for k, v in list(pred_kwargs.items())[1:]:
        fname += f"_{k}_{v}"
    return fname + "_test.pkl"

def compute_metrics(preds):
    TP, FN, FP = computed_confusion_matrix(preds)
    R = TP / (TP + FN) if(TP + FN != 0) else 0
    P = TP / (TP + FP) if(TP + FP != 0) else 0
    

    return R, P, (2*P*R/(P + R) if ((P + R) != 0) else 0)



def computed_confusion_matrix(preds):
    TP = 0
    TPs = []
    FN = 0
    FNs = []
    FP = 0
    FPs = []
    for item in preds:
        titem = item[0]
        
        label_skills = titem["skills"]
        matched_sk = []
        for mskill in titem["matched_skills"]:

            skill_item = titem['matched_skills'][mskill]
            skill_name = skill_item["name+definition"].split(" : ")[0]
            if(skill_name in titem["skills"]):
                TP += 1 ## we predicted a label and indeed in the target ==> TP
                TPs.append([skill_name, titem["skills"]])
            else :
                FP += 1 ## we predicted a lebl as positive but it's false
                FPs.append([skill_name, titem["skills"]])
            matched_sk.append(skill_item)
        for skill in titem["skills"]:
            if(skill != "UNK"):
                if(skill not in matched_sk):
                    FN += 1 ## we predicted a label as not in the skills but it it ==> false negative
                    FNs.append([skill, matched_sk])

    return TP, FN, FP


if __name__ == "__main__":
    main()
            