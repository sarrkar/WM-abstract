import os
import torch
import json
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



def dec2bin(value):
    formatstr = '0%db' % 6
    x =format(value, formatstr)
    return [int(char) for char in x]

def nback_bin_task_index_mapper():
    bin_tmps = []
    nback_bin_task_index_map = np.zeros((3,3)) # each row for 1 n, each column for 1 feature
    for n_back_n_index in range(1,4):
        for feature_index in range(3):
            bin_tmp = dec2bin(int(n_back_n_index + 3 * feature_index))
            bin_tmps.append(bin_tmp)
            nback_bin_task_index_map[n_back_n_index-1, feature_index] = int(''.join(str(x) for x in bin_tmp))
    return nback_bin_task_index_map


def find_nback_task_rule(decbin_task_index, nback_bin_task_index_map):
    
    index = np.where(nback_bin_task_index_map == decbin_task_index)
    return index[0][0] + 1, index[1][0] # return n for nback, feature (0 for location, 1 for object and 2 for category)



def read_data(basepath, mode = "val_angle"):
    "read the recorded model activation file and make adjustments for further analysis"
    "only include val_angle activations"
    path2file = os.path.join(basepath, "activations.pkl")
    json_path = os.path.join(basepath, "config.json")
    df = pd.read_pickle(path2file)
    with open(json_path, 'r') as file:
        info = json.load(file)
#     checkpoint_path = info["loadmodel_path"]
#     checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    task_name = info["taskname"]
    
    try:
        if info["is_RNN"]: network_type = "RNN"
    except: pass

    try: 
        if info["is_GRU"]: network_type = "GRU"
    except: pass
    
    batchsize = info["batch_size"]
    hidden_size = info["hidden_size"]
    seq_len = info["seq_len"]
    
    if "CNN_activation" in df.keys():
        is_CNN_act_recorded = True
    else: is_CNN_act_recorded = False
        
    bin_tmps = []
    if task_name == "nback":
        nback_bin_task_index_map = np.zeros((3,3)) # each row for 1 n, each column for 1 feature
        for n_back_n_index in range(1,4):
            for feature_index in range(3):
                bin_tmp = dec2bin(int(n_back_n_index + 3 * feature_index))
                bin_tmps.append(bin_tmp)
                nback_bin_task_index_map[n_back_n_index-1, feature_index] = int(''.join(str(x) for x in bin_tmp))

    val_angle_df = df[df["mode"] == mode]
    
    # convert task_index from np array to list: np array is not hashable but i need to find all unique task index
    unique_task_indexs = []

    for i in range(len(val_angle_df)):
        if len(unique_task_indexs) == 0:
            is_add = True
        for j in range(len(unique_task_indexs)):
            # if the task_index has been stored, check if next match

            if np.array_equal(val_angle_df.task_index.iloc[i], unique_task_indexs[j]):
                is_add = False # as long as there is one match, do not add new task index
                break
            elif j == len(unique_task_indexs) - 1: # loop to the last one and no matches:
                is_add = True
        if is_add:
            unique_task_indexs.append(list(val_angle_df.task_index.iloc[i]))
            
    ## add additional plain task index column
    plain_task_index = []
    nback_index = []
    feature_index = []
    unique_task_index_map = {}
    nback_bin_task_index_map = nback_bin_task_index_mapper()
    for i in range(len(unique_task_indexs)):
        unique_task_index_map["%d" % i] = unique_task_indexs[i]
#     print("unique task index map")
#     print(unique_task_index_map)
    for i in range(len(val_angle_df)):
        for j in range(len(unique_task_indexs)):
            if np.array_equal(val_angle_df.task_index.iloc[i], unique_task_indexs[j]):
                plain_task_index.append(j)
        if task_name == "nback":
            ni, fi = find_nback_task_rule(int(''.join(str(int(element)) for element in val_angle_df.task_index.iloc[i])),nback_bin_task_index_map)
            nback_index.append(ni)
            feature_index.append(fi)
    assert len(plain_task_index) == len(val_angle_df)
    val_angle_df["plain_task_index"] = plain_task_index
    if task_name == "nback":
        val_angle_df["ntask_index"] = nback_index
        val_angle_df["feature_index"] = feature_index
        
    # confirm equal number of tasks
#     print("Sanity Check! count number of trials per task")
    n_trials_per_task = []
    for i in val_angle_df.plain_task_index.unique():
        l = len(val_angle_df[val_angle_df["plain_task_index"] == i])
#         print("task %d has %d trials" % (i, l))
        n_trials_per_task.append(l)
    plt.bar(np.arange(1, len(n_trials_per_task)+1), n_trials_per_task)
    plt.xlabel("task index")
    plt.ylabel("n_trials")
    
    return task_name, val_angle_df
    
    
# define trial wise accuracy function
def trial_wise_acc(df, task_name):
    "return accuracy for each individual task"
    task_index_list = df.plain_task_index.unique()
    accs = []
    if task_name == "nback":
        ntask_indices = []
        feature_indices = []
    for task_index in task_index_list:
        curr_df = df[df["plain_task_index"] == task_index]
        if task_name == "nback":
            # sanity check: make sure there is only one type of task involved
            assert curr_df.ntask_index.unique().shape[0] == 1
            assert curr_df.feature_index.unique().shape[0] == 1
            ntask_index = curr_df.ntask_index.unique()[0]
            feature_index = curr_df.feature_index.unique()[0]
            ntask_indices.append(ntask_index)
            feature_indices.append(feature_index)

        # calcuate trial wise accuracy
        accs.append(np.sum([(curr_df.predicted_action.iloc[i] == curr_df.corrected_action.iloc[i]).all() for i in range(len(curr_df))])/len(curr_df))
    if task_name == "nback":
         return accs, ntask_indices, feature_indices
    else:
        return accs

# define frame-wise accuracy function
def frame_wise_acc(df, task_name):
    "return framewise accuracy for each individual task"
    task_index_list = df.plain_task_index.unique()
    n_frames = len(df.predicted_action.iloc[0])
    accs = np.empty((len(task_index_list), n_frames))
    if task_name == "nback":
        ntask_indices = []
        feature_indices = []
    for i, task_index in enumerate(task_index_list):
        curr_df = df[df["plain_task_index"] == task_index]
        if task_name == "nback":
            # sanity check: make sure there is only one type of task involved
            assert curr_df.ntask_index.unique().shape[0] == 1
            assert curr_df.feature_index.unique().shape[0] == 1
            ntask_index = curr_df.ntask_index.unique()[0]
            feature_index = curr_df.feature_index.unique()[0]
            ntask_indices.append(ntask_index)
            feature_indices.append(feature_index)

        # calcuate trial wise accuracy
        temp = np.stack([[curr_df.predicted_action.iloc[i] == curr_df.corrected_action.iloc[i]] for i in range(len(curr_df))])
        accs[i,:] = (np.sum(temp, axis = 0)/len(curr_df))[0]
    if task_name == "nback":
         return accs, ntask_indices, feature_indices
    else:
        return accs


def frame_based_corrected_trials(df, frame_check = [1], is_balanced = True):
    "return trials with corrected trials for frames in frame_check, if is_balanced is True, subsample to the smallest trials of all tasks"
    task_index_list = df.plain_task_index.unique()
    update_dfs = [] 
    min_len_df = 100000

    for i, task_index in enumerate(task_index_list):
        curr_df = df[df["plain_task_index"] == task_index]
        selected_trials = [i for i in range(len(curr_df)) if (curr_df.predicted_action.iloc[i][frame_check] == curr_df.corrected_action.iloc[i][frame_check]).all()]
        curr_df = curr_df.iloc[selected_trials]
        update_dfs.append(curr_df)

        if min_len_df > len(curr_df):
            min_len_df = len(curr_df)
    
    if is_balanced: # subsample each df to min_len_df
        for i, curr_df in enumerate(update_dfs):
            curr_df = curr_df.sample(n=min_len_df, random_state=42)
            update_dfs[i] = curr_df

#     print("sanity check: is there equal number of trials for each task?")
#     print([len(curr_df) for curr_df in update_dfs])
    return pd.concat(update_dfs, ignore_index=True)


def concat_nback_decoder_data_gen(RNN_activation,ntask_index, selected_ntask_index, feature_index, selected_feature_index, decoding_feature_labels, split_ratio = 0.8):
    # get [0.8,0.2] splited activation for selected task at frames on frame_list
    indice_1 = np.where(ntask_index == selected_ntask_index)
    indice_2 = np.where(feature_index == selected_feature_index)
    indice = np.intersect1d(indice_1, indice_2)
    # randomize indices
    np.random.shuffle(indice)
    
    curr_data = RNN_activation[indice]
    # subtract the mean of the RNN activation
    curr_data = curr_data - np.mean(curr_data, axis = 0)
    print("curr data shape:", curr_data.shape)
    
    curr_label = decoding_feature_labels[indice]
    l = curr_data.shape[0]
    l_train = int(np.floor(l*split_ratio))

    train_data = curr_data[:l_train]
    val_data = curr_data[l_train:]
    train_label = curr_label[:l_train]
    val_label = curr_label[l_train:]

    return train_data, train_label, val_data, val_label

def nback_decoder_data_gen(df, nback_index, feature_index, decoding_feature, frame_list = [0], split_ratio = 0.8):
    # get [0.8,0.2] splited activation for selected task at frames on frame_list
    curr_df = df[(df["ntask_index"] == nback_index) & (df["feature_index"] == feature_index)]
    curr_df = curr_df.sample(frac=1, random_state=42)
    train_df = curr_df.iloc[:int(np.floor(len(curr_df)*split_ratio))]
    val_df = curr_df.iloc[int(np.floor(len(curr_df)*split_ratio)):]

    n_units = train_df.activation.iloc[0].shape[-1]
    train_data = np.stack(train_df.activation.to_numpy())[:,frame_list,:].reshape(-1, n_units)
    val_data = np.stack(val_df.activation.to_numpy())[:,frame_list,:].reshape(-1, n_units)

    if decoding_feature == 0:
        train_label = np.squeeze(np.stack(train_df.input_loc.to_numpy()))[:, frame_list].reshape(-1)
        val_label = np.squeeze(np.stack(val_df.input_loc.to_numpy()))[:, frame_list].reshape(-1)
    elif decoding_feature == 1:
        train_label = np.squeeze(np.stack(train_df.input_obj.to_numpy()))[:, frame_list].reshape(-1)
        val_label = np.squeeze(np.stack(val_df.input_obj.to_numpy()))[:, frame_list].reshape(-1)
    elif decoding_feature == 2:
        train_label = np.squeeze(np.stack(train_df.input_ctg.to_numpy()))[:, frame_list].reshape(-1)
        val_label = np.squeeze(np.stack(val_df.input_ctg.to_numpy()))[:, frame_list].reshape(-1)


    return train_data, train_label, val_data, val_label


def nback_decoder_cnn_data_gen(df, decoding_feature, frame_list = [0], split_ratio = 0.8, is_pca = True):
    # get [0.8,0.2] splited activation for selected task at frames on frame_list
    curr_df = df
    curr_df = curr_df.sample(frac=1, random_state=42)
    
    n_units = 256*7*7
    data = np.stack(curr_df.CNN_activation_2.to_numpy())[:,frame_list,:].reshape(-1, n_units)

    if decoding_feature == 0:
        label = np.squeeze(np.stack(curr_df.input_loc.to_numpy()))[:, frame_list].reshape(-1)
    elif decoding_feature == 1:
        label = np.squeeze(np.stack(curr_df.input_obj.to_numpy()))[:, frame_list].reshape(-1)
    elif decoding_feature == 2:
        label = np.squeeze(np.stack(curr_df.input_ctg.to_numpy()))[:, frame_list].reshape(-1)
    
    if is_pca:
        # Instantiate a PCA object
        pca = PCA(n_components = 200)

        # Fit the PCA model to your data
        pca.fit(data)

        # Transform the data into the principal components
        transformed_data = pca.transform(data)
        data = transformed_data
        # Explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Print the explained variance ratio
        print("Explained Variance Ratio:", explained_variance_ratio)

        # Plot the cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        plt.figure()
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Principal Components')
        plt.grid()
        plt.show()
    
    
    train_data = data[:int(np.floor(len(curr_df)*split_ratio))]
    val_data = data[int(np.floor(len(curr_df)*split_ratio)):]
    train_label = label[:int(np.floor(len(curr_df)*split_ratio))]
    val_label = label[int(np.floor(len(curr_df)*split_ratio)):]
    
    return train_data, train_label, val_data, val_label




def decoder(train_data, val_data, train_label, val_label, type = "svm_linear", return_weight_vector = False):
    if type == "svm_linear":
        clf = svm.SVC(kernel='linear', C=1)  # Linear kernel with regularization parameter C
    elif type == "svm_rbf":
        clf = svm.SVC(kernel='rbf', C=1, gamma='scale') # rbf kernel
    elif type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=4000, random_state=42)

    # Train the classifier on the training data
    clf.fit(train_data, train_label)
    

    # Get the weight vector and intercept
    weight_vector = clf.coef_
    intercept = clf.intercept_

    # Make predictions on the test data
    val_pred = clf.predict(val_data)
    
    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(val_label, val_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if return_weight_vector:
        return accuracy, weight_vector
    else:
        return accuracy


# for each individual task feature, each nback n task, construct the activation matrix
# and corresponding feature template matrix
def RSA_template_matrix_constructor(df, ntask_index, feature_index, n_samples, method = "corr", is_return_flabel = False):
    ### only consider the first frame for now!!!!!!
    curr_df = df[(df["ntask_index"] == ntask_index) & (df["feature_index"] == feature_index)].sample(n=n_samples)
    RNN_activations = np.stack(curr_df.activation.to_numpy())[:,0,:].reshape(-1, curr_df.activation.iloc[0].shape[-1])
    CNN_activations = np.stack(curr_df.CNN_activation_2.to_numpy())[:,0].reshape(-1, np.prod(curr_df.CNN_activation_2.iloc[0].shape[1:]))

    if method == "corr":
        RNN_corr = np.corrcoef(RNN_activations, rowvar=True)
        CNN_corr = np.corrcoef(CNN_activations, rowvar=True)
    elif method == "Euclidean":
        RNN_corr = scipy.spatial.distance.cdist(RNN_activations, RNN_activations, 'euclidean')
        CNN_corr = scipy.spatial.distance.cdist(CNN_activations, CNN_activations, 'euclidean')
    elif method == "cosine_similarity":
        RNN_corr = sklearn.metrics.pairwise.cosine_similarity(RNN_activations)
        CNN_corr = sklearn.metrics.pairwise.cosine_similarity(CNN_activations)

    loc_index = np.squeeze(np.stack(curr_df.input_loc.to_numpy()))[:,0].reshape(-1)
    obj_index = np.squeeze(np.stack(curr_df.input_obj.to_numpy()))[:,0].reshape(-1)
    ctg_index = np.squeeze(np.stack(curr_df.input_ctg.to_numpy()))[:,0].reshape(-1)

    loc_template = np.array([0 if i != j else 1 for i in loc_index for j in loc_index]).reshape(len(loc_index), len(loc_index))
    obj_template = np.array([0 if i != j else 1 for i in obj_index for j in obj_index]).reshape(len(obj_index), len(obj_index))
    ctg_template = np.array([0 if i != j else 1 for i in ctg_index for j in ctg_index]).reshape(len(ctg_index), len(ctg_index))
    
    if not is_return_flabel:
        return RNN_corr, CNN_corr, loc_template, obj_template, ctg_template
    else:
        return RNN_activations, CNN_activations, loc_index, obj_index, ctg_index, loc_template, obj_template, ctg_template

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)
