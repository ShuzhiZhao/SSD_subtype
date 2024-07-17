import time
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans, KMeans 
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin  
from sklearn.datasets import make_blobs 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import csv

def CH_EvaInd_Vis(X,Y,result_dir):
    if torch.is_tensor(X):
        X = X.detach().numpy()
    if torch.is_tensor(Y):
        Y = Y.detach().numpy()
    Y = Y.reshape((-1))
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False    
    
    # 在实际工作中是人工给定的，专门用于判断聚类的效果的一个值
    ### TODO: 实际工作中，我们假定聚类算法的模型都是比较可以，最多用轮廓系数/模型的score api返回值进行度量；
    ### 其它的效果度量方式一般不用
    ### 原因：其它度量方式需要给定数据的实际的y值 ===> 当我给定y值的时候，其实我可以直接使用分类算法了，不需要使用聚类
    ch_scores = []
    # Looking for Calinski-Harabaz
    star = 3
    for i in range(star,18):
        km = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
        km.fit(X)
        ch_scores.append(metrics.calinski_harabasz_score(X,km.labels_))
    plt.figure(dpi=150)
    plt.plot(range(star,18),ch_scores,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('calinski_harabaz_score')
    plt.savefig(result_dir+'Calinski-Harabaz.tif',dpi=120)
    plt.clf()
    plt.close()

    clusters = ch_scores.index(max(ch_scores))+2
#     print('clusters number:',clusters)

    k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
    t0 = time.time() 
    k_means.fit(X)  
    km_batch = time.time() - t0  
#     print ("K-Means算法模型训练消耗时间:%.4fs" % km_batch)

    batch_size = 100
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)  
    t0 = time.time()  
    mbk.fit(X)  
    mbk_batch = time.time() - t0  
#     print ("Mini Batch K-Means算法模型训练消耗时间:%.4fs" % mbk_batch)

    km_y_hat = k_means.labels_
    mbkm_y_hat = mbk.labels_
#     print(km_y_hat) # 样本所属的类别

    k_means_cluster_centers = k_means.cluster_centers_
    mbk_means_cluster_centers = mbk.cluster_centers_
#     print ("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
#     print ("Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,  
                                      mbk_means_cluster_centers) 

    ### 效果评估
    score_funcs = [
        metrics.adjusted_rand_score,#ARI
        metrics.v_measure_score,#均一性和完整性的加权平均
        metrics.adjusted_mutual_info_score,#AMI
        metrics.mutual_info_score,#互信息
#         metrics.silhouette_score#Silhouette Coefficient
    ]
    # save clusterIndex in CSV    
    header = [
        metrics.adjusted_rand_score.__name__,#ARI
        metrics.v_measure_score.__name__,#均一性和完整性的加权平均
        metrics.adjusted_mutual_info_score.__name__,#AMI
        metrics.mutual_info_score.__name__,#互信息
#         metrics.silhouette_score.__name__#Silhouette Coefficient
    ]
    ## 2. 迭代对每个评估函数进行评估操作
    with open(result_dir+"_k-mean_clusterIndex.csv",'w') as f:
        datas = {}
        for score_func in score_funcs:
            t0 = time.time()
            km_scores = score_func(Y,km_y_hat)
            datas[score_func.__name__] = km_scores
#             print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__,km_scores, time.time() - t0))
        writer = csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerows([datas])
    with open(result_dir+"_MiniBatchk-mean_clusterIndex.csv",'a',newline='',encoding='utf-8') as f:
        datas = {}
        for score_func in score_funcs:
            t0 = time.time()
            mbkm_scores = score_func(Y,mbkm_y_hat)
            datas[score_func.__name__] = mbkm_scores  
#             print("Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__,mbkm_scores, time.time() - t0))
        writer = csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerows([datas])
        
    # Visualization
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(X_tsne[:,0],X_tsne[:,1],c=Y,label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:,0],X_pca[:,1],c=Y,label="PCA")
    plt.legend()
    plt.savefig(result_dir+'Visual.tif',dpi=120)
    plt.clf()
    plt.close()

centers = [[1, 1], [-1, -1], [1, -1]] 
clusters = len(centers)
X, Y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.95, random_state=28) 
result_dir = "/media/lhj/Momery/Microstate_HJL/NS_subtype/Result/cluster/VAE_"
CH_EvaInd_Vis(X,Y,result_dir)