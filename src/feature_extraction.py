""" 
A script for feature extraction.

Used many modified and intact code blocks from 
'https://github.com/jeffrey-palmerino/BugLocalizationDNN'
"""

import nltk
# nltk.download('punkt_tab')
nltk.download('punkt')
# nltk.download('stopwords')

from util import *
from joblib import Parallel, delayed, cpu_count
import csv
import os


def extract(i, br, bug_reports, java_src_dict):
    """ Extracts features for 50 wrong(randomly chosen) files for each
        right(buggy) file for the given bug report.
    
    Arguments:
        i {integer} -- Index for printing information
        br {dictionary} -- Given bug report 
        bug_reports {list of dictionaries} -- All bug reports
        java_src_dict {dictionary} -- A dictionary of java source codes
    """
    print("Bug report : {} / {}".format(i + 1, len(bug_reports)), end="\r")  

    br_id = br["id"]
    br_date = br["report_time"]
    br_files = br["files"]
    br_raw_text = br["raw_text"]
    # print(f"Files in bug report: {br_files}")  # In rõ các file liên quan đến báo lỗi

    features = []

    for java_file in br_files:
        java_file = os.path.normpath(java_file)  # Chuẩn hóa đường dẫn đến file

        try:
            # Kiểm tra nếu file không tồn tại trong java_src_dict
            if java_file not in java_src_dict:
                print(f"Warning: File {java_file} not found in source code dictionary")
                continue

            # Source code của file Java
            src = java_src_dict[java_file]

            # Tính toán độ tương đồng cosine
            rvsm = cosine_sim(br_raw_text, src)
            print(f"Cosine Similarity (rVSM) for {java_file}: {rvsm}")

            # Class Name Similarity
            cns = class_name_similarity(br_raw_text, src)

            # Previous Reports
            prev_reports = previous_reports(java_file, br_date, bug_reports)

            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)

            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)

            # Bug Fixing Frequency
            bff = len(prev_reports)

            features.append([br_id, java_file, rvsm, 1])

            for java_file, rvsm, cns in top_k_wrong_files(
                br_files, br_raw_text, java_src_dict
            ):
                features.append([br_id, java_file, rvsm, 0])

        except:
            # print(f"Error processing")
            pass  # Tiếp tục với file tiếp theo nếu gặp lỗi

    return features


def extract_features():
    """Clones the git repository and parallelizes the feature extraction process
    """

    # Read bug reports from tab separated file
    bug_reports = tsv2dict("/content/drive/MyDrive/GENAI/LLMS/bug-localization/all_data_buglocalization/bug reports/Tomcat.txt")
    #print(bug_reports)
    # Read all java source files
    java_src_dict = get_all_source_code("/content/drive/MyDrive/GENAI/LLMS/bug-localization/all_data_buglocalization/source files/tomcat-7.0.51")
    #print(java_src_dict)

    # Use all CPUs except one to speed up extraction and avoid computer lagging
    batches = Parallel(n_jobs=cpu_count() - 1)(
        delayed(extract)(i, br, bug_reports, java_src_dict)
        for i, br in enumerate(bug_reports)
    )

    # Flatten features
    features = [row for batch in batches for row in batch]
    #print(features)

    # Save features to a csv file
    features_path = os.path.normpath("/content/drive/MyDrive/GENAI/LLMS/bug-localization/all_data_buglocalization/featuresTomcat.csv")
    with open(features_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "report_id",
                "file",
                "rVSM_similarity",
                "match",
            ]
        )
        for row in features:
            writer.writerow(row)


extract_features()