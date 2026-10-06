import json
import os
import pandas as pd
import numpy as np

def prep_data_get_dimensions_numpy(train_path, diffex=False):
    if not "train" in train_path:
        print("Please provide the path to a train dataset as the train_path parameter")
        raise ValueError
    config = {}
    prefix = train_path.split('_train')[0]
    config['name'] = os.path.basename(prefix)
    config['prefix'] = prefix
    config['train_path'] = train_path
    train_features = np.load(train_path)
    train_labels = np.load(train_path.replace('features', 'labels'))

    test_path = train_path.replace('train', 'test')
    test_features = np.load(test_path)
    test_labels = np.load(test_path.replace('features', 'labels'))

    val_path = train_path.replace('train', 'val')
    val_features = np.load(val_path)
    val_labels = np.load(val_path.replace('features', 'labels'))

    if diffex and train_features.shape[1] > 1000: 
        gene_names = pd.read_csv(os.path.join(os.path.dirname(prefix), 'all_genes.csv'), index_col=0, names=['transcript'], header=0).transcript.squeeze().tolist()
        differentially_expressed = pd.read_csv(os.path.join(os.path.dirname(prefix), f"differentially_expressed_{config['name']}.csv"), header=None, names=['transcripts']).squeeze().tolist()
        differentially_expressed = [x for x in gene_names if x in differentially_expressed]
        train_features = pd.DataFrame(train_features, columns=gene_names)[differentially_expressed].values
        test_features = pd.DataFrame(test_features, columns=gene_names)[differentially_expressed].values
        val_features = pd.DataFrame(val_features, columns=gene_names)[differentially_expressed].values
        config['differentially_expressed'] = differentially_expressed

    try: 
        vocabulary_path = prefix + '_mapping.json'
        print(f'loading vocabulary path {vocabulary_path=}')
        with open(vocabulary_path, 'r') as file:
            vocabulary = json.load(file)
        # print(f"{len(vocabulary)=}")
        config['output_dim'] = len(vocabulary)
        print(f"in prep data get dimensions: {config['output_dim']=}")

        config['binary'] = False
        config['vocabulary'] = vocabulary

    except:
        print("No dictionary")

    if config['output_dim'] == 2:
        config['output_dim'] = 1 #assume binary classification
        config['binary'] = True
        train_labels = train_labels[:, None]
        test_labels = test_labels[:, None]
        val_labels = val_labels[:, None]
    
    def _set_dimensions(train_features):

        return np.nan, train_features.shape[1], False, train_features.shape[0]

    batch_size, input_dim, is_categorical, total_samples = _set_dimensions(train_features)
    print(f"{batch_size=}, {input_dim=}, {is_categorical=}, {total_samples=}")
    # print("after set dimensions")
    # print(train.element_spec)
    config['source_batch_size'] = batch_size
    config['input_dim'] = input_dim
    config['total_samples'] = total_samples

    config['train_features'] = train_features
    config['test_features'] = test_features
    config['val_features'] = val_features
    
    config['train_labels'] = train_labels
    config['test_labels'] = test_labels
    config['val_labels'] = val_labels
    return config


def dataset_to_numpy_fixed(tf_dataset):
    features_list = []
    labels_list = []
    
    unbatched_dataset = tf_dataset.unbatch()
    
    for features, labels in unbatched_dataset.as_numpy_iterator():
        features_flat = np.asarray(features).ravel()
        features_list.append(features_flat)
        
        label_array = np.asarray(labels)
        
        if label_array.size == 1:
            class_scalar = label_array.item()  # Handles binary (0 or 1)
        else:
            class_scalar = np.argmax(label_array).item()  # Handles 32-class one-hot
            
        labels_list.append(class_scalar)
        
    X = np.vstack(features_list)
    y = np.array(labels_list)
    
    return X, y
