**DistanceClassifier** is a distance-based classifier based on the sklearn BaseEstimator class. It classifies instances based on how close they are to the centroid of the training instances for each class. It can use Mahalanobis distance or Euclidean distance. 

Theory
===

Given a labeled training set $\mathcal{T} = \{(\mathbf{x}_i,y_i), i = 1\;\dots\;n\}$, consisting of $n$ samples of attributes $\mathbf{x}_i \in \mathbb{R}^p$ associated with the corresponding class label $y_i$ from the set $\mathcal{C} = \{c_1\;\dots\;c_k\}$. The $n \times p$ matrix of attribute samples $\mathbf{X}$ can be partitioned according to its labels into $k$ subsets $\{\mathbf{X}_1\;\dots\;\mathbf{X}_k\}$, such that $\mathbf{X}_j$ is the subset of $\mathbf{\mathbf{X}}$ tagged with class label $c_j$. The Distance Classifier classifies a new sample $\mathbf{x}' \in \mathbb{R}^{p}$ by measuring the distance of $\mathbf{x}'$ to each subset $\{\mathbf{X}_1\;\dots\;\mathbf{X}_k\}$, and then assign the class label corresponding to the minimum distance, i.e.

\begin{equation}\label{eq:disc_fn}
\hat{y}(\mathbf{x}') = c_j, \;\; {\rm if} \;\; j = \arg \min_{\ell} D(\mathbf{x}',\mathbf{X}_{\ell}) \;\; , \; \ell = 1, \dots, k
\end{equation}  
One such measure is the Mahalanobis distance, $D_M$, 
\begin{equation}\label{eq:MD}
D_M(\mathbf{x}',\mathbf{X}_j) = \sqrt{\left(\mathbf{x}'-\mathbf{\mu}_j\right)\mathbf{\Sigma}_{j}^{-1}\left(\mathbf{x}'-\mathbf{\mu}_j\right)^T}
\end{equation}
where $\mathbf{\mu}_j \in \mathbb{R}^p$ is the centroid of $\mathbf{X}_j$ and $\mathbf{\Sigma}_j \in \mathbb{R}^{p \times p}$ is its covariance matrix, rendering $D_M$  the equivalent Euclidean distance of $\mathbf{x}'$ from $\mathbf{X}_j$, scaled by the eigenvalues (variances) and rotated by the eigenvectors of $\mathbf{\Sigma}_j$, to account for the correlation between columns of $\mathbf{X}_j$.
