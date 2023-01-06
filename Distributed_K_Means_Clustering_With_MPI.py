from mpi4py import MPI
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
from collections import Counter

def k_clustering(loc_tf_idf,loc_centroids):
    k,N=loc_centroids.shape[0],loc_tf_idf.shape[0]
    d_main=[]
    for i in range(N):
        d=[np.sqrt((loc_tf_idf[i]-loc_centroids[j]).power(2).sum()) for j in range(k)]
        d_main.append(d)
    d_mat=np.array(d_main)
    membership=(np.argmin(d_mat, axis=1)).reshape((-1,1))
    cluster_l,new_centroid_l=[],[]
    for m in range(k):
        cluster=loc_tf_idf[np.nonzero(membership == m)[0]]
        cluster_l.append(cluster)
        if cluster.shape[0]!=0:
            new_centroid_l.append(cluster.mean(axis=0))
        else:
            new_centroid_l.append(loc_centroids[m].todense())
    return new_centroid_l,membership

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        k = 2
        corpus = fetch_20newsgroups(subset='all')
        data=corpus['data']
        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(data)
        N=tf_idf.shape[0]
        np.random.seed(4)
        C_index=np.random.randint(N, size=k)
        gl_centroids = tf_idf[C_index]
        #splitting indices.
        indices_split=np.array_split(list(range(0,N)),p)
    else:
        indices_split=None
        tf_idf=None
        gl_centroids=None
    time_start = MPI.Wtime()
    indices_split= comm.scatter(indices_split, root=0)
    tf_idf=comm.bcast(tf_idf,root=0)
    tolerance,counter=2,0
    while tolerance>1:
        counter+=1
        gl_centroids=comm.bcast(gl_centroids,root=0)
        new_centroids,new_membership=k_clustering(tf_idf[indices_split],gl_centroids)
        new_membership_l=comm.gather(new_membership,root=0)
        new_centroids_l=comm.gather(new_centroids,root=0)
        if rank == 0:
            new_membership_=np.vstack(new_membership_l)
            new_gl_centroid = []
            for cluster_num in range(k):
                cl_centroid_list = [l[cluster_num] for l in new_centroids_l]
                cl_centroids_vec = np.vstack(cl_centroid_list)
                new_gl_centroid.append(np.mean(cl_centroids_vec, axis=0))
            new_gl_centroid = np.vstack(new_gl_centroid)
            old_centroids = gl_centroids
            gl_centroids = csr_matrix(new_gl_centroid)
            tolerance=abs(gl_centroids-old_centroids).sum()/k
            # CONVERGENCE
            if tolerance <= 1:
                end = MPI.Wtime()
                total_time=round(MPI.Wtime() - time_start, 3)
                print(
                    f"time taken for clustering with processors={p}  and cluster number ={k} is {total_time}")
                with open('time.txt', 'a') as f:
                    f.write(f'{p} {k} {total_time} {counter}\n')
                with open('membership_2.pickle', 'wb') as data:
                    pickle.dump(new_membership_, data)
                # print("*****************************************************")
                # print(f"the centroid matrix of {k} number of centroids is")
                # print(new_gl_centroid)
                # print("*****************************************************")
                # print(f"Total number of members for each cluster is ")
                # new_membership_list=list(new_membership_.reshape((-1,)))
                # print(Counter(new_membership_list))
        else:
            tolerance = None
        tolerance = comm.bcast(tolerance, root=0)

# with open('membership4.pickle', 'rb') as f:
#     membership4 = pickle.load(f)
# with open('membership4.pickle', 'rb') as f:
#     membership1 = pickle.load(f)
# sum_of_diff=np.sum(abs(membership1-membership4))
# print(f"difference between serially and paralelly computed memberships is {sum_of_diff}")


