
from models.k_means.k_means import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def task_3():
    # Load the dataset
    df = pd.read_feather('./data/external/word-embeddings.feather')

    print(df.keys())# ['words','vit']
    #words: 200 words like car,book, .. 
    #vit: 200 corresponding feature vector each (512,) normalized decimal number 

    # print(df['vit'][1])



    # Extract feature vectors
    X = np.vstack(df['vit'].values)


    # Elbow Method to determine optimal number of clusters
    wcss = []
    k_range = range(1, 11) 


    for k in k_range:
        kmeans = KMeans(k=k, max_iter=1000, tol=0.0001) #used kmeans/kmeans.py class
        kmeans.fit(X)
        print(f"Convergence reached at {kmeans.conver}")
        wcss.append(kmeans.getCost(X))



    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(k_range)
    plt.grid(True)
    # plt.savefig("./assignments/2/figures/task3.png")
    # plt.show()


    #Q.  Perform K-means clustering on the dataset using the number of clusters as kkmeans1
    #the optimal number of clusters (k) is determined from the elbow plot
    optimal_k = 6  # Replace with the k value you find optimal
    #k_means1=6
    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(k=optimal_k, max_iter=100, tol=0.0001)
    kmeans.fit(X)

    # Retrieve clustering results
    cluster_centers = kmeans.centroids
    labels = kmeans.predict(X) 

    # Print some results
    print(f'Cluster Centers Shape: {cluster_centers.shape}')
    print(f'Labels Shape: {labels}')

    # (Optional) Print the number of samples in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print('Number of samples in each cluster:', cluster_counts)


# task_3()





from models.gmm.gmm import *
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Load the dataset
df = pd.read_feather('./data/external/word-embeddings.feather')

# Prepare the data (vit embeddings)
X = np.vstack(df['vit'].values)  # X will be a matrix of shape (200, 512) assuming 200 words and 512 features

###
def randomdata():
    np.random.seed(42)

    # Generate data from 3 different Gaussian clusters
    n_samples = 500
    mean1 = [2, 5]
    cov1 = [[0.3, 0.4], [0.4, 0.7]]

    mean2 = [3, 5]
    cov2 = [[0.3, -0.4], [-0.4, 0.7]]

    mean3 = [4, 5]
    cov3 = [[0.3, 0.4], [0.4, 0.7]]

    X1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
    X3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)
    global X
    X = np.vstack([X1, X2, X3]) 
# randomdata()
###


def task_4_1():
    try:
        # Initialize GMM with the number of components (e.g., 3 Gaussians)
        gmm = GMM(3, max_iter=100)
        # Fit the GMM model to the data
        gmm.fit(X)

        # Get the optimal parameters (weights, means, covariances)
        params = gmm.getParams()
        print("GMM Parameters:")
        print("Weights:", params['weights'])
        print("Means:", params['means'])
        # print("Covariances:", params['covariances'])

        # Get the membership values (responsibilities) for each sample
        memberships = gmm.getMembership()
        print("Memberships (responsibilities):")
        print(memberships)

        # Get the overall log-likelihood of the dataset
        likelihood = gmm.getLikelihood(X)
        print("Log-Likelihood of the dataset:", likelihood)

        # plot_gmm_memberships(X, memberships)




    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, add more cleanup or logging here if needed
        return  # Exit the function if an error occurs

    """my implementation faces numerical instability in covariance inversion and log-likelihood calculation, gets likehood inf , in feather dataset only,;in random dataset gets correct like sklearns """
    """reason :  inf/number is scource of this in formula"""

# task_4_1() #gives same likehood like sklearn's for random data, so class working 

def task_4_2_1():
    
    n_components=3
    gmm = GaussianMixture(n_components=n_components, max_iter=100)
    
    # Fit the GMM model to the data
    gmm.fit(X)
    
    # Get the optimal parameters (weights, means, covariances)
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    
    # Get the membership values (responsibilities) for each sample
    responsibilities = gmm.predict_proba(X)
    
    # Get the overall log-likelihood of the dataset
    likelihood = gmm.score(X) * X.shape[0]  # `score` returns log-likelihood per sample
    
    print("GMM Parameters (sklearn):")
    print("Weights:", weights)
    print("Means:", means)
    print("Covariances:", covariances)
    
    print("Memberships (responsibilities):")
    print(responsibilities[:5]) #let's see first 5
    
    print("Log-Likelihood of the dataset:", likelihood)

    # plot_gmm_memberships(X, responsibilities)
    """
    Why This Should Work:
    Robustness: sklearn's GaussianMixture is designed to handle large datasets and high-dimensional data robustly.
    Numerical Stability: It includes internal mechanisms to prevent issues with numerical instability and singularity.
    Automatic Regularization: It automatically regularizes covariance matrices to ensure they are positive semi-definite."""

# task_4_2_1()  



def task_4_2_cont():
    # Define the range of cluster numbers to test
    n_clusters_range = range(1, 11)  # You can adjust this range based on your needs

    bic_scores = []
    aic_scores = []

    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=0,covariance_type="spherical")
        gmm.fit(X)
        
        n_features = X.shape[1] #512
        cov_params = n_clusters  #n_components , covariance_type = spherical
        mean_params = n_features * n_clusters# n_components
        _n_parameters= int(cov_params + mean_params + n_clusters - 1)
        # Calculated BIC and AIC with Formula
        bic=-2 * gmm.score(X) * X.shape[0] + _n_parameters * np.log(X.shape[0])
        # bic_scores.append(gmm.bic(X))
        bic_scores.append(bic)

        aic= -2 * gmm.score(X) * X.shape[0] + 2 * _n_parameters
        # aic_scores.append(gmm.aic(X))
        aic_scores.append(aic)

    # Determine the optimal number of clusters based on BIC and AIC
    optimal_clusters_bic = n_clusters_range[np.argmin(bic_scores)]
    optimal_clusters_aic = n_clusters_range[np.argmin(aic_scores)]

    print(f'Optimal number of clusters based on BIC: {optimal_clusters_bic}')
    print(f'Optimal number of clusters based on AIC: {optimal_clusters_aic}')


    "Report : since aic is min at 2, bic at 9 , at 5 both have similiar and minimum value ,so it take optimal_cluster_value as =5"

    # Plotting the BIC and AIC scores
    fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot BIC scores on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('BIC Score', color=color)
    ax1.plot(n_clusters_range, bic_scores, marker='o', color=color, label='BIC')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for AIC scores
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AIC Score', color=color)
    ax2.plot(n_clusters_range, aic_scores, marker='o', color=color, linestyle='--', label='AIC') 
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Add title and grid
    plt.title('BIC and AIC Scores for Different Number of Clusters')
    fig.tight_layout()
    plt.grid(True)
    # plt.savefig("./assignments/2/figures/task_4_2AIC_BIC.png")
    plt.show()


    "Report : since aic is min at 2, bic at 9 , at 5 both have similiar and minimum value ,so it take optimal_cluster_value as =5"

   
    
    #2ND HALF PART
    # Perform GMM clustering with the optimal number of clusters (kgmm1)
    kgmm1 = 5  # or optimal_clusters_aic, depending on which you prefer

    gmm_best = GaussianMixture(n_components=kgmm1, random_state=0)
    gmm_best.fit(X)

    print("Log Likelihood of GMM:", gmm_best.lower_bound_)


    # Get the GMM parameters and membership
    params_best = {
        'means': gmm_best.means_,
        'covariances': gmm_best.covariances_,
        'weights': gmm_best.weights_
    }
    memberships_best = gmm_best.predict_proba(X)
    print("Memberships (responsibilities):")
    print(memberships_best[:5])

    print("GMM Parameters (Optimal Clusters):")
    print("Weights:", params_best['weights'].shape) #remove shape, to see Parameters
    print("Means:", params_best['means'].shape)
    print("Covariances:", params_best['covariances'].shape)

# task_4_2_cont()







from models.pca.pca import *
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_feather('./data/external/word-embeddings.feather')



# Extract the word embeddings
X = np.array(df['vit'].tolist())  # Assuming 'vit' column contains the embeddings


def task_5_1():
    # Initialize the PCA with the desired number of components
    n_components = 3  # For example, reduce to 2 dimensions
    pca = PCA(n_components=n_components)

    # Fit the PCA model
    pca.fit(X)

    # Transform the data
    X_pca = pca.transform(X)

    # Check if PCA is working correctly
    is_valid = pca.checkPCA()
    print(f"PCA valid: {is_valid}")

    # Optional: print the transformed data
    print("Transformed data:")
    print(X_pca)

# task_5_1()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def task_5_2():
    # Initialize PCA for 2D
    pca_2d = PCA(n_components=2)

    # Fit and transform the data to 2D
    pca_2d.fit(X)
    X_pca_2d = pca_2d.transform(X)

    # Check if PCA for 2D is working correctly
    is_valid_2d = pca_2d.checkPCA()
    print(f"PCA 2D valid: {is_valid_2d}")

    # Initialize PCA for 3D
    pca_3d = PCA(n_components=3)

    # Fit and transform the data to 3D
    pca_3d.fit(X)
    X_pca_3d = pca_3d.transform(X)

    # Check if PCA for 3D is working correctly
    is_valid_3d = pca_3d.checkPCA()
    print(f"PCA 3D valid: {is_valid_3d}")

    # Plotting the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, c='blue', edgecolor='k')
    plt.title('PCA 2D Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('./assignments/2/figures/pca_2d_projection.png')  # Save the 2D plot to a file
    plt.show()

    # Plotting the 3D projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.5, c='green', edgecolor='k')
    ax.set_title('PCA 3D Projection')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig('./assignments/2/figures/pca_3d_projection.png')  # Save the 2D plot to a file
    plt.show()


task_5_2()
"""considering images, there are 3 clusters. 1 cluster if considered few as outlier"""



def task_5_3():
    pca_2d = PCA(n_components=2)

    # Fit and transform the data to 2D
    pca_2d.fit(X)
    X_pca_2d = pca_2d.transform(X)

    # Print the principal components
    print("Principal Component 1:", pca_2d.components[:, 0]) #shape (512,)
    print("Principal Component 2:", pca_2d.components[:, 1])

    # Plotting the 2D PCA Projection
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, c='blue', edgecolor='k')
    plt.title('PCA 2D Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('./assignments/2/figures/pca_2d_projection.png')  # Save the 2D plot to a file
    plt.show()

    # Initialize PCA for 3D
    pca_3d = PCA(n_components=3)
    # Fit and transform the data to 3D
    pca_3d.fit(X)
    X_pca_3d = pca_3d.transform(X)

    print("Principal Component 1:", pca_3d.components[:, 0])
    print("Principal Component 2:", pca_3d.components[:, 1])
    print("Principal Component 3:", pca_3d.components[:, 2])

    # Plotting the 3D projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.5, c='green', edgecolor='k')
    ax.set_title('PCA 3D Projection')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig('./assignments/2/figures/pca_3d_projection.png')  # Save the 2D plot to a file
    plt.show()

    """k2 = 3 clusters in 2D, and 3D plot """

# task_5_3()

























import pandas as pd
import numpy as np
from models.pca.pca import *
from models.k_means.k_means import *
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_feather('./data/external/word-embeddings.feather')

def task_6_1():

    # Extract the word embeddings
    X = np.array(df['vit'].tolist())  # Assuming 'vit' column contains the embeddings

    #2D visualization
    k2=3
    pca = PCA(n_components=k2)
    pca.fit(X)
    X_reduced= pca.transform(X)

    kmeans = KMeans(k=3)
    kmeans.fit(X_reduced)
    # Predict cluster labels
    kmeans_labels = kmeans.predict(X_reduced)

    # Plot the clustering results
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6)
    plt.title('K-means Clustering')
    plt.savefig('./assignments/2/figures/task_6_1.png')
    plt.show()

# task_6_1()

##############################33

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Reduced_dataset=None

def task_6_2():

    # Load the dataset
    df = pd.read_feather('./data/external/word-embeddings.feather')

    # Extract the word embeddings
    X = np.array(df['vit'].tolist())  # Assuming 'vit' column contains the embeddings

    # Initialize PCA
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)

    # Get eigenvalues (variance of each component)
    eigenvalues = pca.explained_variance_

    # Calculate the cumulative variance explained
    cumulative_variance = np.cumsum(eigenvalues)
    total_variance = np.sum(eigenvalues)

    # Find the number of components needed to explain 90% of the variance ;Slides
    threshold = 0.9 * total_variance
    optimal_num_components = np.argmax(cumulative_variance >= threshold) + 1

    print(f'Number of components needed to explain 90% of the variance: {optimal_num_components}')
    # Plot Scree Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalues (Variance)')
    plt.grid(True)
    plt.savefig('./assignments/2/figures/scree_plot.png')
    plt.show()
    #made SCREE Plot and found optimal number=107

    pca= PCA(optimal_num_components)
    pca.fit(X)
    reduced_dataset=pca.transform(X) #got reduced datased
    
    global Reduced_dataset
    Reduced_dataset=reduced_dataset

    # Elbow Method to determine optimal number of clusters
    wcss = []
    k_range = range(1, 11) 


    for k in k_range:
        kmeans = KMeans(k=k, max_iter=1000, tol=0.0001) #used kmeans/kmeans.py class
        kmeans.fit(reduced_dataset)
        print(f"Convergence reached at {kmeans.conver}")
        wcss.append(kmeans.getCost(reduced_dataset))



    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(k_range)
    plt.grid(True)
    # plt.savefig("./assignments/2/figures/task_6_2.png")
    # plt.show()

    #PERFORMING KMEANS ON REDUCED DATASED WITH K_means3
    print("reduced dataset shape ",reduced_dataset.shape)
    k_means3 =9 #value from elbow plot

    kmeans= KMeans(k=k_means3,max_iter=1000,tol=0.0001)
    kmeans.fit(reduced_dataset)

    cluster_centers = kmeans.centroids
    labels= kmeans.predict(reduced_dataset)

    # Print some results
    print(f'Cluster Centers Shape: {cluster_centers.shape}')
    print(f'Labels Shape: {labels}')

    # (Optional) Print the number of samples in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print('Number of samples in each cluster:', cluster_counts)

# task_6_2()



from models.gmm.gmm import *
def task_6_3():
    df = pd.read_feather('./data/external/word-embeddings.feather')    # Extract the word embeddings
    X = np.array(df['vit'].tolist())  # Assuming 'vit' column contains the embeddings
    k2=3

    gmm = GMM(k2, max_iter=100)
    gmm.fit(X)
    memberships = gmm.getMembership()
    params = gmm.getParams()
    print("GMM Parameters:")
    print("Weights:", params['weights'])
    print("Means:", params['means'])
    print("Covariances:", params['covariances'])
    print("Memberships (responsibilities):")
    print(memberships)

# task_6_3()





# task 6_4
from sklearn.mixture import GaussianMixture
from models.gmm.gmm import * #used in last part;reqired
def task_6_4():    

    task_6_2()
    Reduced_dataset #;updated after calling task_6_2()


    n_clusters_range = range(1, 11)  # You can adjust this range based on your needs

    bic_scores = []
    aic_scores = []

    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=0,covariance_type="spherical")
        gmm.fit(Reduced_dataset)
        
        # Calculate BIC and AIC
        bic_scores.append(gmm.bic(Reduced_dataset))
        aic_scores.append(gmm.aic(Reduced_dataset))

    # Determine the optimal number of clusters based on BIC and AIC
    optimal_clusters_bic = n_clusters_range[np.argmin(bic_scores)]
    optimal_clusters_aic = n_clusters_range[np.argmin(aic_scores)]

    print(f'Optimal number of clusters based on BIC: {optimal_clusters_bic}')
    print(f'Optimal number of clusters based on AIC: {optimal_clusters_aic}')

    # Plotting the BIC and AIC scores
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot BIC scores on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('BIC Score', color=color)
    ax1.plot(n_clusters_range, bic_scores, marker='o', color=color, label='BIC')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for AIC scores
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AIC Score', color=color)
    ax2.plot(n_clusters_range, aic_scores, marker='o', color=color, linestyle='--', label='AIC') 
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Add title and grid
    plt.title('BIC and AIC Scores for Different Number of Clusters')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("./assignments/2/figures/task_6_4.png")
    # plt.show()

    "REPORT:"
    "BIC: More conservative and aims to find a simpler model that explains the data."
    "AIC: Aims to find a model that has a good fit to the data without being overly complex."
    "AIC: A lower AIC value indicates a model that has a good fit to the data with less complexity."
    "BIC: A lower BIC value also indicates a simpler, better-fitting model.  "
    "So i took k_best (aka k_gmm3)  =5 "
    k_gmm3= 5 # is best value representing min in aic,bic plot
    Reduced_dataset

    print(Reduced_dataset.shape)
    gmm = GMM(k_gmm3, max_iter=100)
    gmm.fit(Reduced_dataset)
    params = gmm.getParams()
    print("GMM Parameters:")
    print("Weights:", params['weights'])
    print("Means:", params['means'])
    print("Covariances:", params['covariances'])
    memberships = gmm.getMembership()
    print("Memberships (responsibilities):")
    print(memberships)
    # Get the overall log-likelihood of the dataset
    likelihood = gmm.getLikelihood(Reduced_dataset)
    print("Log-Likelihood of the dataset:", likelihood) #57500.457713263706
    


# task_6_4()












k_means1=6
k2 =3
k_means3 =9

k_gmm1 =5
k2     =3
k_gmm3 =5


from models.k_means.k_means import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def task_7_1():
    df = pd.read_feather('./data/external/word-embeddings.feather')

    print(df.keys())
    # print(df['words'])
    # Prepare the data (vit embeddings)
    print("K_means1 :")
    X = np.vstack(df['vit'].values) 
    kmeans = KMeans(k=k_means1, max_iter=100, tol=0.0001)
    kmeans.fit(X)
    labels = kmeans.predict(X) 


    # Add the cluster labels to the DataFrame
    df['cluster'] = labels

    # Print words grouped by cluster labels
    for cluster in range(k_means1):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()

  ########################
    
    df = pd.read_feather('./data/external/word-embeddings.feather')

    print(df.keys())
    # print(df['words'])
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values) 
    kmeans = KMeans(k=k2, max_iter=100, tol=0.0001)
    kmeans.fit(X)
    labels = kmeans.predict(X) 


    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    print("K2 :")
    # Print words grouped by cluster labels
    for cluster in range(k2):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()

#######################################
    df = pd.read_feather('./data/external/word-embeddings.feather')

    print(df.keys())
    # print(df['words'])
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values) 
    kmeans = KMeans(k=k_means3, max_iter=100, tol=0.0001)
    kmeans.fit(X)
    labels = kmeans.predict(X) 


    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    print("K_means3 :")
    # Print words grouped by cluster labels
    for cluster in range(k_means3):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()


# task_7_1()




import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture  # Import GMM from sklearn

def task_7_2():
    # Load the dataset
    df = pd.read_feather('./data/external/word-embeddings.feather')
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values)

    # Define the number of clusters (replace with the number of clusters you need)
    n_clusters = k_gmm1  # Example number of clusters
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100, tol=0.0001)
    gmm.fit(X)

    # Predict the cluster labels for each word
    labels = gmm.predict(X)

    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    print("K_gmm1 :")
    # Print words grouped by cluster labels
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()


    #####################################################
    df = pd.read_feather('./data/external/word-embeddings.feather')
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values)

    # Define the number of clusters (replace with the number of clusters you need)
    n_clusters = k2  # Example number of clusters
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100, tol=0.0001)
    gmm.fit(X)

    # Predict the cluster labels for each word
    labels = gmm.predict(X)

    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    # Print words grouped by cluster labels
    print("K2 :")
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()


    #############################################
    df = pd.read_feather('./data/external/word-embeddings.feather')
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values)

    # Define the number of clusters (replace with the number of clusters you need)
    n_clusters = k_gmm3  # Example number of clusters
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100, tol=0.0001)
    gmm.fit(X)

    # Predict the cluster labels for each word
    labels = gmm.predict(X)

    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    # Print words grouped by cluster labels
    print("K_gmm3 :")
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()


# task_7_2()





def task_7_3():
    df = pd.read_feather('./data/external/word-embeddings.feather')
    kmmean =6
    print(df.keys())
    # print(df['words'])
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values) 
    kmeans = KMeans(k=kmmean, max_iter=100, tol=0.0001)
    kmeans.fit(X)
    labels = kmeans.predict(X) 


    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    print("K_means3 :")
    # Print words grouped by cluster labels
    print("kmmean =6")
    for cluster in range(kmmean):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()


    #############################

    df = pd.read_feather('./data/external/word-embeddings.feather')
    # Prepare the data (vit embeddings)
    X = np.vstack(df['vit'].values)
    kgmm = 5
    # Define the number of clusters (replace with the number of clusters you need)
    n_clusters = kgmm  # Example number of clusters
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100, tol=0.0001)
    gmm.fit(X)

    # Predict the cluster labels for each word
    labels = gmm.predict(X)

    # Add the cluster labels to the DataFrame
    df['cluster'] = labels
    # Print words grouped by cluster labels
    print("kgmm = 5 ")
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        cluster_words = df[df['cluster'] == cluster]['words'].tolist()
        print(cluster_words)
        print()

# task_7_3()








import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
kmmean = 6
kgmm = 5
# Load the dataset
df = pd.read_feather('./data/external/word-embeddings.feather')

X = df['vit'].tolist()  # Assuming 'vit' column contains the 512-dimensional vectors


from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

linkage_methods = ['single', 'complete', 'average', 'ward']
distance_metric = 'euclidean'

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, method in enumerate(linkage_methods):
    Z = hierarchy.linkage(X, method=method, metric=distance_metric)
    hierarchy.dendrogram(Z, ax=axes[i], truncate_mode='lastp', p=15, leaf_rotation=90)
    axes[i].set_title(f'{method.capitalize()} Linkage')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Distance')

plt.tight_layout()
plt.savefig("./assignments/2/figures/task8_2_eucledian")
# plt.show()


def task_8_2():
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Load the dataset
    df = pd.read_feather('./data/external/word-embeddings.feather')
    X = df['vit'].tolist()  # Assuming 'vit' column contains 512-dimensional vectors

    # Define linkage methods and distance metrics to experiment with
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'cosine']  # Ward's method requires 'euclidean' distance only

    # Function to plot dendrogram for a specific linkage method and distance metric
    def plot_dendrogram(X, method, metric):
        Z = linkage(X, method=method, metric=metric)
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title(f'Dendrogram - Linkage: {method}, Metric: {metric}')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.savefig(f"./assignments/2/figures/task8_2_{method}_{metric}.png")
        # plt.show()

    # Loop through each combination of linkage method and distance metric
    for method in linkage_methods:
        for metric in distance_metrics:
            # Ward's method requires Euclidean distance, skip other metrics for 'ward'
            if method == 'ward' and metric != 'euclidean':
                continue
            plot_dendrogram(X, method, metric)




# task_8_2()



import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
def task_8_3():

    # TASK 3
    # FINDA BEST LINKAGE METHOD ,FROM ABOVE
    #  Using the Euclidean distance metric

    # Load the dataset
    df = pd.read_feather('./data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)  # Assuming 'vit' column contains 512-dimensional vectors

    kmmean = 6
    kgmm = 5
    kbest1, kbest2 = kmmean, kgmm

    # Compute the linkage matrix using the best linkage method and Euclidean distance
    Z = linkage(X, method='ward', metric='euclidean')  #'ward' is the best method based on analysis

    # Plot the dendrogram for visualization
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram (Linkage: ward, Metric: Euclidean)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig("./assignments/2/figures/task8_3.png")
    # plt.show()

    # Cut the dendrogram to form clusters using kbest1 and kbest2
    clusters_kbest1 = fcluster(Z, kbest1, criterion='maxclust')  # Clusters corresponding to kbest1 from K-Means
    clusters_kbest2 = fcluster(Z, kbest2, criterion='maxclust')  # Clusters corresponding to kbest2 from GMM

    # Compare with K-Means clustering
    kmeans = KMeans(n_clusters=kbest1, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Compare with GMM clustering
    gmm = GaussianMixture(n_components=kbest2, random_state=42)
    gmm_labels = gmm.fit_predict(X)

    # Function to compare cluster assignments
    def compare_clusters(labels1, labels2, method_name1, method_name2):
        match = np.sum(labels1 == labels2)
        total = len(labels1)
        match_percentage = (match / total) * 100
        print(f"Cluster alignment between {method_name1} and {method_name2}: {match_percentage:.2f}%")

    # Compare hierarchical clustering vs K-Means
    compare_clusters(clusters_kbest1, kmeans_labels, "Hierarchical Clustering (kbest1)", "K-Means")

    # Compare hierarchical clustering vs GMM
    compare_clusters(clusters_kbest2, gmm_labels, "Hierarchical Clustering (kbest2)", "GMM")

# task_8_3()














import pandas as pd
import numpy as np
from models.pca.pca import *
from models.k_means.k_means import *
import matplotlib.pyplot as plt
from models.knn.knn import *
import time
# Load the dataset
columns = [
    'duration_ms','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'track_genre'
]
df = pd.read_csv(r'./data/external/spotify.csv', usecols=columns)

genres = df['track_genre']


# Select feature columns (excluding one-hot encoded genre columns for now)
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature'
] 

def task_9_1():
    X = df[features]
    print(X.shape)


    # Step 4: Apply PCA
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)


    # Get explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio()
    print("Explained Variance Ratio:", explained_variance_ratio)

    # Get cumulative explained variance for the scree plot
    cumulative_variance = np.cumsum(explained_variance_ratio)


    # Generate scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    plt.title('Scree Plot: Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig("./assignments/2/figures/task_9_1.png")
    plt.show()



    

    optimal_number_of_dimension=5
    pca= PCA(optimal_number_of_dimension)
    pca.fit(X)
    reduced_dataset=pca.transform(X) #based on optimal principal component

    # Append genre labels to the reduced dataset
    reduced_dataset_with_genre = np.column_stack((reduced_dataset, genres))


    #  Use the KNN model implemented in Assignment 1 on the reduced dataset using the best {k, distance metric} pair obtained
    # k: 15, Distance Metric: Manhattan, Prediction Type: Weighted Sum



    # Split the data into train, validation, and test sets
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(reduced_dataset_with_genre)  # Shuffle the data

    reduced_dataset_with_genre = reduced_dataset_with_genre[:70000]

    total_size = len(reduced_dataset_with_genre)
    train_size = int(0.8 * total_size)
    test_size = val_size = int(0.1 * total_size)

    train_data = reduced_dataset_with_genre[:train_size]
    temp_data = reduced_dataset_with_genre[train_size:]
    test_data = temp_data[:test_size]
    val_data = temp_data[test_size:]

    # Create DataFrames for the reduced dataset and genre column
    n_components = reduced_dataset.shape[1]
    columns = [f'PC{i+1}' for i in range(n_components)] + ['track_genre']

    train_df = pd.DataFrame(train_data, columns=columns)
    test_df = pd.DataFrame(test_data, columns=columns)
    val_df = pd.DataFrame(val_data, columns=columns)

    # Save to CSV
    train_df.to_csv('./data/interim/1/reduced_spotify_train.csv', index=False)
    test_df.to_csv('./data/interim/1/reduced_spotify_test.csv', index=False)
    val_df.to_csv('./data/interim/1/reduced_spotify_val.csv', index=False)



    # Load the split datasets
    train_df = pd.read_csv('./data/interim/1/reduced_spotify_train.csv')
    test_df = pd.read_csv('./data/interim/1/reduced_spotify_test.csv')
    val_df = pd.read_csv('./data/interim/1/reduced_spotify_val.csv')

    Y_train = train_df['track_genre'].values
    Y_val = val_df['track_genre'].values

    X_train = train_df[["PC1","PC2","PC3","PC4","PC5"]].values
    X_val = val_df[["PC1","PC2","PC3","PC4","PC5"]].values

    

    knn = KNearestNeighbours(k=15, distance_metric='manhattan', prediction_type="weighted_sum")
    knn.fit(X_train, Y_train)
    a=time.time()
    metrics = knn.validate(X_val, Y_val)
    print("metrices ",metrics)
    print("time taken ",time.time()-a)

# task_9_1() # also part of 9_2 metrices are printed for doen here ,this after scree plot, optimal components, reduced dataset
"metrices  {'macro_precision': 0.11141427381482807, 'macro_recall': 0.11455254643344683, 'macro_f1': 0.11104979622419338, 'micro_precision': 0.11508771929824561, 'micro_recall': 0.11508771929824561, 'micro_f1': 0.11508771929824561, 'accuracy': 0.9844752231455832}"
# time taken  79.51951432228088  on 114002 full dataset
# time taken  30.119790077209473 on 70000 samples
# time taken  13.056809663772583 on 50000 samples








def task_9_2():
    # Load the split datasets
    train_df = pd.read_csv('./data/interim/1/spotify_train.csv')
    val_df = pd.read_csv('./data/interim/1/spotify_val.csv')

    # Extract target variables
    Y_train = train_df['track_genre'].values
    Y_val = val_df['track_genre'].values
    print(Y_train.shape)

    # Initialize hyperparameters
    pred_type = ['weighted_sum']#weighted_sum
    

    X_train = train_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature']].values
    X_val = val_df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature']].values
    knn = KNearestNeighbours(k=15, distance_metric='manhattan', prediction_type=pred_type[0])
    knn.fit(X_train, Y_train)
    a= time.time()
    print("metrices ",knn.validate(X_val, Y_val))
    print("time taken ",time.time()-a)
    

task_9_2() # contains all column , asign 1 to change samples
# metrices  {'macro_precision': 0.18147587154189257, 'macro_recall': 0.18053097639379498, 'macro_f1': 0.17786395033004604, 'micro_precision': 0.1794736842105263, 'micro_recall': 0.1794736842105263, 'micro_f1': 0.1794736842105263, 'accuracy': 0.9856048014773776}
# time taken  98.41144156455994 on 114002 full dataset ,
# time taken  34.277494192123413   70000 samples
# time taken  17.290382146835327   on 50000 samples



import matplotlib.pyplot as plt
def plot_reducedvsfull():
    

    # Data
    dataset_sizes = [50000, 70000, 114002]
    time_reduced = [13.056809663772583, 30.119790077209473, 79.51951432228088]
    time_full = [17.290382146835327, 34.277494192123413, 98.41144156455994]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, time_reduced, marker='o', linestyle='-', color='blue', label='Reduced Dataset')
    plt.plot(dataset_sizes, time_full, marker='o', linestyle='-', color='red', label='Full Dataset')

    # Add labels and title
    plt.xlabel('Dataset Size')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time for KNN Model on Complete and Reduced Datasets')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.xscale('log')  # Optional: if dataset sizes vary greatly, logarithmic scale might be helpful
    plt.savefig("./assignments/2/figures/task_9_2_plot")
    plt.show()

# plot_reducedvsfull()

