# Assignment 1 Report

SMAI REPORT :2023121006¬¬¬¬¬¬¬¬
The figures folder has been moved into the assignment-specific directory. You should save all the generated plots, animations, etc inside that folder and then include them in the report.
Run : python -m assignments.1.a1 --data_sample {is_integer_total_data_rows_for_knn}

## KNN

Task_2_2_1()

Distribution :

![alt text](image.png)

Observation: time_signature, mode are not distribted. Hence we wont be able to put vector completely on basis of such feature.
These features don't vary much across your dataset, they might not contribute significantly to the KNN model's ability to differentiate between genres.

Acousticness, danceability,valence, energy should be most relevant for genre classification as they are more widespread in space.
Because they capture more nuanced characteristics of the audio, they are likely to be more useful for genre classification

![alt text](image-1.png)

Plot of order/hierarchy according to most relevant features using spearman correlation

### Task :2_3_1 done in models/knn/knn.py and performance_measures/performance.py took care

<b>Note :
Used concurrent.futures to paralleize multiple possibilities for {k,distance_metric} ,finding best pair.
Parallelism not used/done in knn.py
</b>

### Uncomment : task3_1_3()

1. Find the best {k, distance metric} pair that gives the best validation accuracy for an 80:10:10 split (train:test:validation).
   Ans: I permuted dis_metric=['euclidean','manhattan','cosine','hamming'] and pred_type= [ 'most_common','weighted_sum'] ,k=[1,3,5,7,9,11,13,15,17,19] . Took ‘hamming’ and ‘weighted_sum’ as extra to experiment with

2. Print an ordered rank list of the top 10 such pairs.
   Top 10 Accuracy Pairs:
   k: 15, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985706
   k: 17, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985702
   k: 19, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985676
   k: 13, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985645
   k: 11, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985639
   k: 9, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985594
   k: 7, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985565
   k: 5, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985532
   k: 19, Distance Metric: Euclidean, Prediction Type: Weighted Sum, Accuracy: 0.985491
   k: 1, Distance Metric: Manhattan, Prediction Type: Weighted Sum, Accuracy: 0.985489

### 3. Plot k vs accuracy given a choice (yours) of any given {k, distance metric} pair with a constant data split. (on full data):

heatmap to represent best way: for most_common and weighted_sum
![alt text](image-2.png)

![alt text](image-3.png)

### task3_4_5()

4. More data need not necessarily mean best results. Try dropping various columns and check if you get better accuracy. Document the combination with which you get the best results.\
   <b>WITH :k: 15, Distance Metric: Manhattan, Prediction Type: Weighted Sum<\b>
   \

#### Top 5 results:

features ('danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo') , accuracy 0.9858387196060326
features ('danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature') , accuracy 0.9858156355801785  
 features ('danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo') , accuracy 0.9857602339181286
features ('danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature') , accuracy 0.98567405355494  
 features ('danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence') , accuracy 0.9855263157894737

### 5. [Bonus] Try for all combinations of columns and see which combination gives the best result. See if that matches with the analysis made during data exploration.

This combination :('danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo')
Yes , it matches with the analysis made during data exploration.
Since , all top features in hierarchy result is obtained

### 2.5 Optimization

1. Improve the execution time of the program using vectorization.
   Done
2. Plot inference time for initial KNN model, best KNN model, most optimized KNN model, and the default sklearn KNN model.
   Run :python -m assignments.1.a1 --data_sample 10000
   • Sklearn KNN accuracy 0.145, time 1.6473705768585205
   • New KNN accuracy 0.145, time 23.533416748046875 (vectorized /most optimized)
   • Old KNN accuracy 0.145, time 28.056403636932373 (original /not optimized)

![alt text](image-4.png)
<T>For 10000 data samples</T>

</br>
### task_2_5_3()
3. Plot the inference time vs train dataset size for initial KNN model, best KNN model, most optimized KNN model, and the default sklearn KNN model. Write down observations that you make based on this graph.</br>
•	dataset_sizes =  ([100, 500, 1000, 5000, 10000, 50000, 114000])</br>
•	old_knn_times =([0.00275, 0.06090, 0.23844, 6.88793, 24.84275, 591.44909, 3029.85131])</br>
•	new_knn_times =([0.00234, 0.06524, 0.21555, 6.87011, 17.39268, 120.09435, 314.80442])</br>
•	sklearn_knn_times =([0.00225, 0.00583, 0.20349,  1.230361, 1.99732, 3.39953, 5.46194])</br>
since there is large difference between values, used log/log scale

![alt text](image-5.png)

## task_2_6()

### 2.6 Second Dataset:

There is another dataset given in data/external/spotify-2, which is already split into train.csv, validate.csv and test.csv. Using the best {k, distance metric} pair you got previously, apply KNN on this data and state your observations on the data and the performance of this data.

• WITH :k: 15, Distance Metric: Manhattan, Prediction Type: Weighted Sum

Result :
measures {'macro_precision': 0.1501911362929458, 'macro_recall': 0.1510026846327197, 'macro_f1': 0.14675549714933445, 'micro_precision': 0.15, 'micro_recall': 0.15, 'micro_f1': 0.15, 'accuracy': 0.9850877192982456}

Accuracy is same as obtained by sklearn with best ‘weighted sum’ and on full dataset

### Linear Regression

Converges before max_epcoh if {np.all(np.abs(new_coefficients - self.coefficients) < self.tolerence=1e-6)
}

## 3.1 Simple Regression:

Shuffle the data and make an 80:10:10 split into train, validation and test. Report metrics for all three sets

## Task_3_1_0()

• Training Metrics:
• Variance (Train): 1.0862730512494836
• Standard Deviation (Train): 1.042244237810641

• Validation Metrics:
• Variance (Validation): 1.7663210794374133
• Standard Deviation (Validation): 1.3290301273625866

• Test Metrics:
• Variance (Test): 1.1421742596393099
• Standard Deviation (Test): 1.0687255305452892

## 3.1.1 Degree 1: Fit a line to the curve (y = β1x + β0). Report the MSE, standard deviation and variance metrics on train and test splits. Plot the training points with the fitted line

    task_3_1_1()
    Tried Learning Rate=	[ 0.15, 0.2, 0.4,0.01, 0.05, 0.1]
    Found Best learning rate: 0.4 with convergence at epoch 42 (best convergence)

At LR=0.4
Training Metrics:
Variance (Train): 1.0862730512494836
Standard Deviation (Train): 1.042244237810641

Test Metrics:
Variance (Validation): 1.1421742596393099
Standard Deviation (Validation): 1.0687255305452892
MSE test: 0.395541447387086

Line fitted on training data set

![alt text](image-6.png)

### 3.1.2 Degree > 1:

Fit a polynomial to the curve (k degree polynomial). Make a class for regression that can take the value of k as a parameter. Test it with various values of k and report the MSE, standard deviation and variance metrics on train and test splits for all the values. Additionally, report the k that minimizes the error on the test set.

### task_3_1_2()

    tried on : k_lis = [ 1,2,3,4,5,6,7,11,20] , LR =0.1 (having >1 is not good)
    variance ,Standard deviation on y_pred ,mse on (y_pred,y_true)
    Plot made on X_train,Y_train

• k 1 has mse : 0.3955414029030607
• Variance: 0.7460429480989158
• Standard Deviation: 0.8637377773948038
• Converged at epoch: 166
• Train
• k 1 has mse : 0.3149730231382185
• Variance: 0.771284583633458 \
• Standard Deviation: 0.8782280931702526

• k 2 has mse : 0.20317969263986754
• Variance: 0.8270867024717325
• Standard Deviation: 0.9094430726943454
• Converged at epoch: 566
• Train
• k 2 has mse : 0.21549201166424767
• Variance: 0.8707656524653974
• Standard Deviation: 0.9331482478499316
![alt text](image-7.png)

• k 3 has mse : 0.060824748056261826\
• Variance: 1.0236171653396209\
• Standard Deviation: 1.0117396727121166\
• Converged at epoch: 2846

• k 4 has mse : 0.06311303501129592\
• Variance: 1.0196011028895975\
• Standard Deviation: 1.0097529910278045\
• Converged at epoch: 9580

• k 5 has mse : 0.028743733799014286\
• Variance: 1.0920579814814189\
• Standard Deviation: 1.0450157804939688\
• Converged at epoch: 10000\
• Train\
• k 5 has mse : 0.02548918675316326\
• Variance: 1.0483414944892424\
• Standard Deviation: 1.0238854889533509\

![alt text](image-8.png)

• k 6 has mse : 0.029537844201620665\
• Variance: 1.075899855204575\
• Standard Deviation: 1.037255925605911\
• Converged at epoch: 10000\

• k 7 has mse : 0.023211131805735443\
• Variance: 1.098654630570873\
• Standard Deviation: 1.0481672722284707\
• Converged at epoch: 10000\

• k 11 has mse : 0.019132435360465265\
• Variance: 1.0705867312241004\
• Standard Deviation: 1.0346916116525253\
• Train\
• k 11 has mse : 0.009315990301252153\
• Variance: 1.0764663948275004\
• Standard Deviation: 1.0375289850541527\

• Converged at epoch: 10000 \
![alt text](image-9.png)

• k 20 has mse : 0.017839436488157945\
• Variance: 1.063176519313167\
• Standard Deviation: 1.031104514253122\
• Converged at epoch: 10000\
• Train\
• k 20 has mse : 0.008738410686143219\
• Variance: 1.0774580440492785\
• Standard Deviation: 1.0380067649342553\

Train :Best k that minimizes error 20 with mse 0.008738410686143219

Test :Best k that minimizes error 20 with mse 0.017839436488157945

## • Complexity vs. Performance:

### • As the degree kkk increases, the model becomes more complex, leading to lower test MSE but increasing the risk of overfitting, especially noticeable in higher degrees like k=11k = 11k=11 and k=20k = 20k=20.

### • The variance and standard deviation do not vary drastically with kkk, but there is a slight increase, indicating that higher-degree models capture more variability in the data.

## • Optimal Degree Consideration:

• For a balance between training accuracy and generalization, degrees around k=5k = 5k=5 to k=7k = 7k=7 might offer a good compromise. These models have reasonably low MSE and don't show extreme divergence between train and test performance.

### • Potential Overfitting:

• At higher degrees (k=11k = 11k=11 and k=20k = 20k=20), the model shows very low training MSE but a comparatively higher test MSE, suggesting that while these models fit the training data exceptionally well, they may not generalize as effectively.

## 3.1.3 Animation: For each iteration, plot the original data along with the line you are fitting to it, as well as the MSE, standard deviation and variance

    task_3_1_3()
    for k in klis= [1,2,3,4,5,6,7,9]

Stored in assignments/1/figures/gif/convergence_k{k}.gif
K=1

# For Gif , contination of report see DOCX report
