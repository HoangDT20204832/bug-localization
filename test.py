# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# import csv
# import re
# import os
# import random
# import timeit
# import string
# import numpy as np
# from datetime import datetime
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer

# def stem_tokens(tokens):
#     """ Remove stopword and stem

#     Arguments:
#         tokens {list} -- tokens to stem 
#     """
#     stemmer = PorterStemmer()
#     removed_stopwords = [
#         stemmer.stem(item) for item in tokens if item not in stopwords.words("english")
#     ]

#     return removed_stopwords

# def normalize(text):
#     """ Lowercase, remove punctuation, tokenize and stem

#     Arguments:
#         text {string} -- A text to normalize
#     """
#     remove_punc_map = dict((ord(char), None) for char in string.punctuation)
#     removed_punc = text.lower().translate(remove_punc_map)
#     tokenized = word_tokenize(removed_punc)
#     stemmed_tokens = stem_tokens(tokenized)

#     return stemmed_tokens

# text1 = "machine learning is fun"
# text2 = "machine learning is challenging"

# # Tính TF-IDF
# # vectorizer = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")
# # tfidf = vectorizer.fit_transform([text1, text2])
# # sim = ((tfidf * tfidf.T).toarray())[0, 1]
# # # Tính cosine similarity trực tiếp
# # #sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
# # print(sim)  # Kết quả: 0.517...
# def tsv2dict(tsv_path):
#     reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
#     dict_list = []
#     for i, line in enumerate(reader):
#         if(i >= 2): break
#         line["files"] = [
#             os.path.normpath(f)
#             for f in line["files"].strip().split()
#             if f.endswith(".java")
#         ]
#         line["raw_text"] = line["summary"] + line["description"]
#         # line["summary"] = clean_and_split(line["summary"][11:])
#         # line["description"] = clean_and_split(line["description"])
#         line["report_time"] = datetime.strptime(
#             line["report_time"], "%Y-%m-%d %H:%M:%S"
#         )

#         dict_list.append(line)
#     return dict_list

# bug_reports = tsv2dict("D:/Me-hi/20242/Phan_mem_use_LLM/test/bug-localization-by-dnn-and-rvsm/Data_bug/AspectJ.txt")
# print(bug_reports)
# Đọc file txt và thay thế "Eclipse SWT" thành "Eclipse_SWT"
with open("D:/Me-hi/20242/Phan_mem_use_LLM/test/bug-localization-by-dnn-and-rvsm/Data_bug/SWT.txt", "r") as file:
    content = file.read()

# Thay thế chuỗi
content = content.replace("Eclipse SWT", "Eclipse_SWT")

# Ghi lại file
with open("D:/Me-hi/20242/Phan_mem_use_LLM/test/bug-localization-by-dnn-and-rvsm/Data_bug/AspectJ.txt", "w") as file:
    file.write(content)

print("Đã thay thế xong!")
