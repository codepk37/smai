import numpy as np
from collections import defaultdict, Counter #at vooting
from performance_measures.performance import *
# ##########
# #call class performance-measures send Y_truth,Y_pred -> 
# ##########

class Old_KNearestNeighbours: #campusx 
    def __init__(self,k,distance_metric,prediction_type='most_common',r=3): #r of minkowski 
        self.k=k
        self.distance_metric=distance_metric
        self.prediction_type = prediction_type  # 'most_common' or 'weighted_sum'
        self.r=r

    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train
        # print("training done")

        if self.distance_metric == 'mahalanobis': #needed for this
            self.covariance_matrix = np.cov(X_train.T)
            self.inv_cov_matrix = np.linalg.inv(self.covariance_matrix)
        # print("Training done")


    def validate(self, X_val, Y_val):
        print("val")
        predictions = self.predict(X_val)
        print("val_done")
        metrices=performance_metrices(Y_val,predictions)
        
        # print(f"Accuracy: {metrices['accuracy']}")
        # print(f"Macro Precision: {metrices['macro_precision']}")
        # print(f"Macro Recall: {metrices['macro_recall']}")
        # print(f"Macro F1 Score: {metrices['macro_f1']}")
        # print(f"Micro Precision: {metrices['micro_precision']}")
        # print(f"Micro Recall: {metrices['micro_recall']}")
        # print(f"Micro F1 Score: {metrices['micro_f1']}")     
        
        return metrices


    def test(self, X_test, Y_test):
        predictions = self.predict(X_test)
        
        metrices=performance_metrices(Y_test,predictions)
        
        # print(f"Accuracy: {metrices['accuracy']}")
        # print(f"Macro Precision: {metrices['macro_precision']}")
        # print(f"Macro Recall: {metrices['macro_recall']}")
        # print(f"Macro F1 Score: {metrices['macro_f1']}")
        # print(f"Micro Precision: {metrices['micro_precision']}")
        # print(f"Micro Recall: {metrices['micro_recall']}")
        # print(f"Micro F1 Score: {metrices['micro_f1']}")     
        
        return metrices


    def predict(self, X_test):
        # predictions = [self.predict_sam(x_test) for x_test in X_test]
        predictions = []
        for x_test in X_test:
            # print("x_test:", x_test)  # Print x_test before making the prediction
            pred = self.predict_sam(x_test)
            predictions.append(pred)
        return np.array(predictions)



    def predict_sam(self, x_test):
        distances = [self.compute_distance(x_test, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        if self.prediction_type == 'most_common':
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            return most_common
        elif self.prediction_type == 'weighted_sum':
            weights = [1 / (d + 1e-8) for d in k_nearest_distances]  # Avoid division by zero
            label_weights = defaultdict(float)
            for label, weight in zip(k_nearest_labels, weights):
                label_weights[label] += weight
            predicted_label = max(label_weights, key=label_weights.get)
            return predicted_label
    
    def compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            if self.r>1000: #case of considered as infinity
                return  np.max(np.abs(x1-x2))
            return np.sum(np.abs(x1 - x2) ** self.r) ** (1 / self.r) #not infinity case
        elif self.distance_metric == 'mahalanobis':
            diff = x1 - x2
            return np.sqrt(np.dot(np.dot(diff.T, self.inv_cov_matrix), diff))
        elif self.distance_metric=='hamming':
            return  np.sum(x1!=x2)
        elif self.distance_metric=='cosine':
            norm_x1 = np.sqrt(np.sum(x1 ** 2))
            norm_x2 = np.sqrt(np.sum(x2 ** 2))
            cosine_similarity = np.dot(x1, x2) / (norm_x1 * norm_x2)
            return 1 - cosine_similarity
        else:
            raise ValueError("Unsupported distance metric")
   



import numpy as np
from collections import defaultdict, Counter
from performance_measures.performance import *

class KNearestNeighbours:
    def __init__(self, k, distance_metric, prediction_type='most_common', r=3):
        self.k = k
        self.distance_metric = distance_metric
        self.prediction_type = prediction_type  # 'most_common' or 'weighted_sum'
        self.r = r

    def fit(self, X_train, Y_train):
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train

        if self.distance_metric == 'mahalanobis':
            self.covariance_matrix = np.cov(X_train.T)
            self.inv_cov_matrix = np.linalg.inv(self.covariance_matrix)

    def validate(self, X_val, Y_val, batch_size=100):
        print("val")
        predictions = self.predict(X_val, batch_size=batch_size)
        print("val_done")
        print("accuracy : ",np.mean(predictions==Y_val))
        metrics = performance_metrices(Y_val, predictions)
        return metrics

    def test(self, X_test, Y_test, batch_size=100):
        predictions = self.predict(X_test, batch_size=batch_size)
        metrics = performance_metrices(Y_test, predictions)
        return metrics

    def predict(self, X_test, batch_size=100):
        n_samples = X_test.shape[0]
        predictions = []

        for i in range(0, n_samples, batch_size):
            X_batch = X_test[i:i + batch_size].astype(np.float32)
            batch_predictions = self._predict_batch(X_batch)
            predictions.extend(batch_predictions)
        # print("all 100 combined")
        return np.array(predictions)

    def _predict_batch(self, X_batch):
        n_batch_samples = X_batch.shape[0]
        batch_predictions = []

        for j in range(n_batch_samples):
            x_test = X_batch[j:j+1, :]  # Processing one test point at a time

            if self.distance_metric == 'euclidean':
                distances = self._euclidean_distance(x_test)
            elif self.distance_metric == 'manhattan':
                distances = self._manhattan_distance(x_test)
            elif self.distance_metric == 'minkowski':
                distances = self._minkowski_distance(x_test)
            elif self.distance_metric == 'mahalanobis':
                distances = self._mahalanobis_distance(x_test)
            elif self.distance_metric == 'hamming':
                distances = self._hamming_distance(x_test)
            elif self.distance_metric == 'cosine':
                distances = self._cosine_distance(x_test)
            else:
                raise ValueError("Unsupported distance metric")

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.Y_train[k_indices]

            if self.prediction_type == 'most_common':
                prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            elif self.prediction_type == 'weighted_sum':
                k_nearest_distances = distances[k_indices]
                weights = 1 / (k_nearest_distances + 1e-8)
                label_weights = defaultdict(float)
                for idx in range(self.k):
                    label_weights[k_nearest_labels[idx]] += weights[idx]
                prediction = max(label_weights, key=label_weights.get)

            batch_predictions.append(prediction)

        return batch_predictions

    def _euclidean_distance(self, x_test):
        return np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))

    def _manhattan_distance(self, x_test):
        return np.sum(np.abs(self.X_train - x_test), axis=1)
    # def _manhattan_distance(self, X_test):
    # # Broadcasting to compute distances between each test sample and all training samples
    #     return np.sum(np.abs(self.X_train[None, :] - X_test[:, None]), axis=2)


    def _minkowski_distance(self, x_test):
        if self.r > 1000:  # Considered as infinity
            return np.max(np.abs(self.X_train - x_test), axis=1)
        return np.sum(np.abs(self.X_train - x_test) ** self.r, axis=1) ** (1 / self.r)

    # def _mahalanobis_distance(self, x_test):
    #     diff = self.X_train - x_test
    #     return np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_cov_matrix, diff))

    def _hamming_distance(self, x_test):
        return np.sum(self.X_train != x_test, axis=1)

    def _cosine_distance(self, x_test):
        # Compute the norm of each training sample
        norm_x1 = np.sqrt(np.sum(self.X_train ** 2, axis=1))  # (91200,)

        norm_x2 = np.sqrt(np.sum(x_test ** 2))  #  (2.2044)


        # Compute the dot product between each training sample and the test sample
        dot_product = np.dot(self.X_train, x_test.T) 
        # print(f"{self.X_train.shape} ,  {x_test.T.shape} , {dot_product.shape}") (91200, 12) ,  (12, 1) , (91200, 1)
        # Compute cosine similarity
        c= norm_x1 * norm_x2 ## (91200,)
        
        cosine_similarity = dot_product.flatten() / (c+1e-8)  # Shape: (n_train_samples,)
        # Compute cosine distance
        return 1 - cosine_similarity

