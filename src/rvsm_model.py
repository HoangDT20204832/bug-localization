from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy, mean_reciprocal_rank, mean_average_precision
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def rsvm_model():
    
    samples = csv2dict("/content/drive/MyDrive/GENAI/LLMS/bug-localization/all_data_buglocalization/featuresTomcat.csv")
    rvsm_list = [float(sample["rVSM_similarity"]) for sample in samples]

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples, True)

    acc_dict = topk_accuarcy(bug_reports, sample_dict, br2files_dict)
    mrr = mean_reciprocal_rank(bug_reports, sample_dict, br2files_dict)
    map_score = mean_average_precision(bug_reports, sample_dict, br2files_dict)

    return acc_dict, mrr, map_score