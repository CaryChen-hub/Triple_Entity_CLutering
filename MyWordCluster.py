# MyClusterWord（簇中心查找改进）
import os
import random
import time
import gensim
import math
import json
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import *
from gensim.models import word2vec

entityDictPath ="./source/entity.txt" #实体地址
cluster_center_file_name = "./source/cluster/center.txt" #存放簇心词
cluster_result_file_path = "./source/cluster/" # 每一簇聚类结果存放文件夹地址
AllTriplePath = "./source/triple.txt" #需要聚类的三元组地址
model = word2vec.Word2Vec.load('./source/Word60.model') #聚类模型
ClusterNum = 5 #三元组聚类个数


# 写文件
def write_file(write_file_path, write_str):
    f = open(write_file_path, 'a', encoding='utf-8')
    f.write(write_str + '\n')
    f.close()

# 读文件
def read_file(read_file_path):
    f = open(read_file_path, 'r', encoding='utf-8')
    read_str = f.read()
    file_str = read_str.split('\n')  # 按回车分割存入list
    return file_str

# 交换
def swap(a, b):
    return b, a

# 创建文件夹
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print
        path + ' 创建成功'
        return path
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return path

# 按当前词和簇心词相似度排序（低->高）,对应索引也随着相似度值一起排序
def bubble_sort_by_similarity(arr_similarity, correspond_index):
    end_index = len(arr_similarity) - 1
    cnt = end_index
    while cnt:
        for j in range(0, end_index):
            if arr_similarity[j] > arr_similarity[j+1]:
                arr_similarity[j], arr_similarity[j+1] = swap(arr_similarity[j], arr_similarity[j+1])
                correspond_index[j], correspond_index[j+1] = swap(correspond_index[j], correspond_index[j+1])
        end_index = end_index - 1
        cnt = cnt - 1
    return arr_similarity, correspond_index

#得到簇心
def get_cluster_center():
    center_word = []
    # 获取第一个簇的簇心
    txt_read = read_file(entityDictPath) #txt读取
    #txt_read = json.load(open(entityDictPath, "r"))["itos"] #jison读取
    word_sum_cnt = len(txt_read)
    first_cluster_center = random.randint(0, word_sum_cnt - 1)
    center_word.append(txt_read[first_cluster_center])
    # 获取其他k-1个簇心
    for cur_word in txt_read:
        cluster_center_index = 0
        while cluster_center_index < len(center_word):
            # 当前词和簇心词相似度计算
            cur_word_similar_with_other_center = model.wv.similarity(cur_word, center_word[cluster_center_index])
            if cur_word_similar_with_other_center > 0.1:  # 当前词和已确定簇心词相似度大于0.25则为当前词不可能为簇心词
                break
            else:
                cluster_center_index = cluster_center_index + 1
        if cluster_center_index == len(center_word):
            center_word.append(cur_word)
        if len(center_word) == ClusterNum:  # 聚类个数
            break
    f_write_cluster_center = open(cluster_center_file_name, 'a', encoding='utf8')
    for i in center_word:
        f_write_cluster_center.write(i + '\n')
    return center_word

#根据簇心聚类
def cluster_by_center_word(center_word):
    original_data = read_file(entityDictPath) #txt读取
    #original_data = json.load(open(entityDictPath, "r"))["itos"] #jison读取
    # 遍历待聚类的词，和簇心词一一计算相似度，放入最相似的簇心词对应的TXT
    for current_word in original_data:
        print(current_word)
        similar_value = []
        correspond_index = []
        cnt = 1  # 每个簇心词对应存放的TXT，从1开始是因为0.txt存的是所有簇心
        if current_word not in center_word:
            for cur_center_word in center_word:
                similar = model.wv.similarity(current_word, cur_center_word)
                similar_value.append(float(similar))
                correspond_index.append(int(cnt))
                cnt = cnt + 1
            # 将当前待处理词和所有簇心词计算相似度后按相似度排序（低->高）
            similar_value, correspond_index = bubble_sort_by_similarity(similar_value, correspond_index)
            index = len(correspond_index) - 1
            nxt_cluster = True  # 相似度最高的类已饱和，查看下一个类
            while nxt_cluster and index > -1:
                cur_max_similar_center_index = correspond_index[index]
                cur_cluster_txt_name = cluster_result_file_path + str(cur_max_similar_center_index)+'/entity.txt'
                read_data = read_file(cur_cluster_txt_name)
                max_n = len(original_data) / len(center_word) + len(original_data) % len(center_word)

                # 相似度最高的类未饱和，将当前词聚入此类
                if len(read_data) < (max_n + 1):
                    write_file(cur_cluster_txt_name, current_word)
                    nxt_cluster = False
                else:
                    index = index - 1  # 相似度最高的类饱和，查看相似度次高者

# 找到带有该实体的三元组生成子空间
def get_triple(EntityPath,AllTriplePath,ResultPath):
    f1=open(EntityPath, "r", encoding='utf-8')
    f2=open(AllTriplePath,'r', encoding='utf-8')
    f = open(ResultPath,'w', encoding='utf-8')
    tripledata = f2.readlines()
    data = f1.readlines()
    i=0
    for line in tripledata:
        triple = line.strip().split(' ')
        for entity in data:
            if str(triple[0]+'\n') == str(entity):
                f.write(triple[0] + ' ' + triple[1] + ' ' + triple[2] + '\n')
                i = i + 1
            elif str(triple[2]+'\n') == str(entity):
                f.write(triple[0] + ' ' + triple[1] + ' ' + triple[2] + '\n')
                i = i + 1
    print(i)
    f1.close()
    f2.close()
    f.close()

# 生成子空间里的实体、关系列表
def generateDict(dataPath, dictSaveDir):
    if type(dataPath) == str:
        print("INFO : Loading standard data!")
        rawDf = pd.read_csv(dataPath,
                            sep=" ",
                            header=None,
                            names=["head", "relation", "tail"],
                            keep_default_na=False,
                            encoding="utf-8")
    elif type(dataPath) == list:
        print("INFO : Loading a list of standard data!")
        rawDf = pd.concat([pd.read_csv(p,
                                       sep=" ",
                                       header=None,
                                       names=["head","relation","tail"],
                                       keep_default_na=False,
                                       encoding="utf-8") for p in dataPath], axis=0)
        rawDf.reset_index(drop=True, inplace=True)

    headCounter = Counter(rawDf["head"])
    tailCounter = Counter(rawDf["tail"])
    relaCounter = Counter(rawDf["relation"])

    # Generate entity and relation list
    entityList = list((headCounter + tailCounter).keys())
    relaList = list(relaCounter.keys())

    # Transform to index dict
    print("INFO : Transform to index dict")
    entityDict = dict([(word, ind) for ind, word in enumerate(entityList)])
    relaDict = dict([(word, ind) for ind, word in enumerate(relaList)])

    # Save path
    entityDictPath = os.path.join(dictSaveDir, "entityDict.json")
    relaDictPath = os.path.join(dictSaveDir, "relationDict.json")

    # Saving dicts
    json.dump({"stoi": entityDict, "itos": entityList}, open(entityDictPath, "w"))
    json.dump({"stoi": relaDict, 'itos': relaList}, open(relaDictPath, "w"))

def main():
    center_word = get_cluster_center() #得到ClusterNum个聚类中心
    cnt = 0
    # 创建ClusterNum个文件夹代表ClusterNum个聚类
    while cnt < len(center_word):
        cur_cluster_path = cluster_result_file_path+str(cnt + 1)
        cur_cluster_txt_name = mkdir(cur_cluster_path)+'/entity.txt'
        write_file(cur_cluster_txt_name, center_word[cnt])
        cnt = cnt + 1

    #根据聚类中心聚类实体
    cluster_by_center_word(center_word)

    #根据实体聚类三元组
    cnt = 0
    while cnt < ClusterNum:
        EntityPath = cluster_result_file_path+str(cnt + 1)+'/entity.txt'
        ResultPath = cluster_result_file_path+str(cnt + 1)+'/AfterTriple.txt'
        get_triple(EntityPath, AllTriplePath,ResultPath)
        cnt = cnt + 1


# 开始计时
start_time = time.time()
if __name__ == '__main__':
    main()

end_time = time.time()  # 结束计时
# 计算聚类耗时
print(('cluster word time cost', end_time-start_time, 's'))
