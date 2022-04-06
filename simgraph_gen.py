import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import porterStemmer as ps
import collections
from itertools import combinations
import json


sim_method = sys.argv[1]
collect_path = sys.argv[2]
out_name = sys.argv[3]
folder_names = os.listdir(collect_path)
print(type(folder_names))


def tokenize_string(string):
    delims = [' ',",",".",":",";","'","\"","@","#","+","!","_","~","&","*","%","^","=","`","|","$","\n","(",")",">","<"]
    for delim in delims:
        string = string.replace(delim," ")
    return string.split()

def  jaccard_Sim(doc1,doc2):
    intersect = list(set(doc1) & set(doc2))
    union = list(set().union(doc1,doc2))
    sim = len(intersect)/len(union)
    return sim


if sim_method == 'jaccard' :
    ps1 = ps.PorterStemmer()
    docs_terms_map = {}
    for folder in folder_names:
        fold_path = os.path.join(collect_path,folder)
        doc_names = os.listdir(fold_path)
        for doc in doc_names:
            doc_id = os.path.join(folder,doc)
            full_path = os.path.join(collect_path,doc_id)            
            file_content = open(full_path,"rb").read().decode(errors='ignore')
            tokens = [ps1.stem(term.lower(),0,len(term) -1) for term in tokenize_string(file_content)]
            tokens= list(set(tokens))
            doc_id = folder + '/' + doc
            docs_terms_map[doc_id] = tokens

    
    with open(out_name,'w+') as outFile:
        docids = list(docs_terms_map.keys())
        unq_doc_pairs = list(combinations(docids, 2))
        # print(unq_doc_pairs[0:30])
        # iterN = 0
        for pair in unq_doc_pairs:
            doc1 = pair[0]
            doc2 = pair[1]
            # print(pair)
            # iterN += 1
            # if(iterN % 50000 == 0):
            #     print(iterN)
            outFile.write(doc1 + ' ' + doc2 + ' ')
            # outFile.write(' ')
            # outFile.write(doc2)
            # outFile.write(' ')
            sim_score = jaccard_Sim(docs_terms_map[doc1],docs_terms_map[doc2])
            outFile.write(str( float("%0.4f" % (sim_score))))
            outFile.write('\n')


def idf_calc(docs_dict,vocab_list,collect_size):
    idf_dict = {}
    for term in vocab_list:
        # print(term)
        freq = 0
        for docterms in docs_dict.values():
            if term in docterms:
                freq = freq + 1
        
        idf_dict[term] = np.log2(1 + collect_size/freq)
    
    return idf_dict

# def cosine_Sim(doc1,doc2):
#     d1_vect = np.array(doc1)
#     d2_vect = np.array(doc2)
#     dot_prod = np.dot(d1_vect,d2_vect)
#     # print('dot',dot_prod)
#     # print('----')
#     # print((np.linalg.norm(d1_vect))*(np.linalg.norm(d2_vect)))
#     score  = dot_prod/((np.linalg.norm(d1_vect))*(np.linalg.norm(d2_vect)))
#     return score

def cos_Sim(d1_tfidf_dict,d2_tfidf_dict):
    common_terms = list(set(d1_tfidf_dict.keys()) & set(d2_tfidf_dict.keys()))
    dot_prod = 0
    for term in common_terms:
      dot_prod += d1_tfidf_dict[term]*d2_tfidf_dict[term]
    d1_vect = np.array(list(d1_tfidf_dict.values()))
    d2_vect = np.array(list(d2_tfidf_dict.values()))
    d1_norm = np.linalg.norm(d1_vect)
    d2_norm = np.linalg.norm(d2_vect)

    return dot_prod/(d1_norm*d2_norm)

if sim_method == 'cosine' :
    ps1 = ps.PorterStemmer()
    docs_terms_map = {}
    vocab = []
    collection_size = 0
    for folder in folder_names:
        fold_path = os.path.join(collect_path,folder)
        doc_names = os.listdir(fold_path)
        for doc in doc_names:
            collection_size += 1
            doc_id = os.path.join(folder,doc)
            full_path = os.path.join(collect_path,doc_id)            
            file_content = open(full_path,"rb").read().decode(errors='ignore')
            all_terms = [ps1.stem(term.lower(),0,len(term) -1) for term in tokenize_string(file_content)]            
            vocab = list(set().union(vocab,all_terms))
            doc_id = folder + '/' + doc
            docs_terms_map[doc_id] = all_terms
            # print(all_terms)

    vocab = sorted(vocab)
    # print('vocab size', len(vocab))
    # print(collection_size)
    idfs_dict = idf_calc(docs_terms_map,vocab,collection_size)
    # with open('myidfs.json','w+') as f:
    #     json.dump(idfs_dict,f)
    
    # idfs_dict = {}
    # with open('myidfs.json','r') as f:
    #     idfs_dict = json.load(f)
        

    doc_tfidfs = {}

    for docid in docs_terms_map.keys():
        doc_all_terms = docs_terms_map[docid]
        # print("all_terms", doc_all_terms)
        unq_terms_tfs= dict(collections.Counter(doc_all_terms))
        # print("freq dict",unq_terms_tfs)
        loc_doc_vect = {}
        for term,freq in unq_terms_tfs.items():
            tf = np.log2(1+freq)
            idf = idfs_dict[term]
            loc_doc_vect[term] = tf*idf
        
        doc_tfidfs[docid] = loc_doc_vect
        # print(docid)
    

    with open(out_name,'w+') as outFile:
        docids = list(doc_tfidfs.keys())
        unq_doc_pairs = list(combinations(docids, 2))
        # print(unq_doc_pairs[0:30])
        for pair in unq_doc_pairs:
            doc1 = pair[0]
            doc2 = pair[1]
            # print(pair)
            #print(doc1)
            # print(doc2)
            # doc1_vect = dict.fromkeys(vocab,0)
            doc1_termDict = doc_tfidfs[doc1]
            #print(doc1_termDict)
            # for term,freq in doc1_termDict.items():
            #     doc1_vect[term] = freq
            
            # doc2_vect = dict.fromkeys(vocab,0)
            doc2_termDict = doc_tfidfs[doc2]
            # for term,freq in doc2_termDict.items():
            #     doc2_vect[term] = freq
            

            # doc1_vect = list(doc1_vect.values())
            # doc2_vect = list(doc2_vect.values())
            #  print(doc1_vect)
            # print(len(doc1_vect))
            outFile.write(doc1 + ' ' + doc2 + ' ')
            
            sim_score = cos_Sim(doc1_termDict,doc2_termDict)
                
            outFile.write(str( float("%0.4f" % (sim_score))) + '\n')
            # outFile.write('\n')
        
               

            





    
