import json
import os
import tensorflow as tf
import pandas as pd
import numpy as np

def prep_data_get_dimensions(train_path, for_MLP=False):
    if not "train" in train_path:
        print("Please provide the path to a train dataset as the train_path parameter")
        raise ValueError
    config = {}
    prefix = train_path.split('_train')[0]
    config['name'] = os.path.basename(prefix)
    config['prefix'] = prefix
    config['train_path'] = train_path
    train = tf.data.Dataset.load(train_path, compression="GZIP")
    test = tf.data.Dataset.load(train_path.replace('train', 'test'), compression="GZIP")
    val = tf.data.Dataset.load(train_path.replace('train', 'val'), compression="GZIP")
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
                # 2. Create the initializer that reads your file line by line
        keys = list(vocabulary.keys())
        values = list(vocabulary.values())

        if "[UNK]" in keys:
            print("Found [UNK]")
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=keys,
                values=values,
                key_dtype=tf.string,
                value_dtype=tf.int64
            )
    
            table = tf.lookup.StaticVocabularyTable(
                initializer, 
                num_oov_buckets=1
            )
        else:
            print("No [UNK]")
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=keys, 
                values=values, 
                key_dtype=tf.string, 
                value_dtype=tf.int64
            )

            # Use StaticHashTable instead of StaticVocabularyTable
            table = tf.lookup.StaticHashTable(initializer, default_value=-1)
            
        config['table'] = table

    except:
        print("No dictionary")

    if config['output_dim'] == 2:
        config['output_dim'] = 1 #assume binary classification
        config['binary'] = True
        if train.element_spec[1].dtype.is_integer:
            print("Labels are already mapped to integers. Skipping lookup.")
        else:
            print("Mapping to binary...")
            # Wrap the integer cast or lookup inside a tf.cond or check beforehand
            train = train.map(lambda x, y: (x, [config['table'].lookup(y)]))
            test = test.map(lambda x, y: (x, [config['table'].lookup(y)]))
            val = val.map(lambda x, y: (x, [config['table'].lookup(y)]))

        # print("after mapping to binary")
        # print(train.element_spec)

    
    def _set_dimensions(train):
        original_batches = tf.data.experimental.cardinality(train).numpy()
        for batch in train.take(1):
            features, labels = batch
            batch_size, input_dim = features.shape

            total_samples = original_batches * batch_size
            steps_per_epoch = int(total_samples // batch_size)
        
            is_categorical = (
                isinstance(labels, pd.Categorical) or 
                pd.api.types.is_categorical_dtype(labels) or 
                np.issubdtype(np.asarray(labels).dtype, np.object_) or 
                np.issubdtype(np.asarray(labels).dtype, np.str_)
            )
            break
        return batch_size, input_dim, is_categorical, total_samples

    batch_size, input_dim, is_categorical, total_samples = _set_dimensions(train)
    print(f"{batch_size=}, {input_dim=}, {is_categorical=}, {total_samples=}")
    # print("after set dimensions")
    # print(train.element_spec)
    config['source_batch_size'] = batch_size
    config['input_dim'] = input_dim
    config['total_samples'] = total_samples

    # def force_flat_unbatch(features, labels):
    #     # We manually split and flatten the tensors right out of the generator
    #     flat_features = tf.data.Dataset.from_tensor_slices(features)
    #     flat_labels = tf.data.Dataset.from_tensor_slices(tf.reshape(labels, [-1]))
    #     return tf.data.Dataset.zip((flat_features, flat_labels))

    # # Apply the separation fix to completely destroy the (1, None) and (None,) bugs
    # train = train.flat_map(force_flat_unbatch)
    # test = test.flat_map(force_flat_unbatch)
    # val = val.flat_map(force_flat_unbatch)

    # batch_size, input_dim, is_categorical, total_samples = _set_dimensions(test)
    # print(f"{batch_size=}, {input_dim=}, {is_categorical=}, {total_samples=}")
    # config['source_batch_size'] = batch_size
    # config['input_dim'] = input_dim
    # config['total_samples'] = total_samples

    train = train.unbatch()
    test = test.unbatch()
    val = val.unbatch()

    for a,b in train.batch(1).take(1):
        print()
        print("Sanity check")
        print(a[0].numpy().shape,b[0].numpy())
    
    def _preprocess(features, labels, n_classes=config['output_dim'], config=config):
        labels_tensor = tf.convert_to_tensor(labels)
        labels_tensor = tf.cast(labels_tensor, tf.float32)

        # Use Python logic driven by your config instead of tf.cond.
        # This keeps the TensorShape fully known and static for Keras!
        if n_classes == 1:
            # Explicitly force a 2D matrix shape of (None, 1)
            # This handles both batched and unbatched binary input safely
            # labels_tensor = tf.reshape(labels_tensor, [-1, 1])
            # labels_tensor = tf.ensure_shape(labels_tensor, [None, 1])
            if not for_MLP:
                labels_tensor = tf.where(labels_tensor == 0, -1.0, 1.0)
            labels_tensor = tf.reshape(labels_tensor, [1]) 
            labels_tensor = tf.ensure_shape(labels_tensor, [1])
        else:
            # For one-hot datasets, lock the final dimension to n_classes
            labels_tensor = tf.reshape(labels_tensor, [n_classes])
            labels_tensor = tf.ensure_shape(labels_tensor, [n_classes])

        return features, labels_tensor




    # print(f"in preprocess: {config['output_dim']=}")

    # config['test'] = test

    # def _preprocess(features, labels, n_classes=config['output_dim'], feature_dim=input_dim):
    #     features_tensor = tf.convert_to_tensor(features)
    #     labels_tensor = tf.convert_to_tensor(labels)

    #     # Keep features as a clean 1D vector
    #     features_tensor = tf.reshape(features_tensor, [feature_dim])
    #     features_tensor = tf.ensure_shape(features_tensor, [feature_dim])

    #     if n_classes == 1:
    #         # FIX: Use tf.squeeze to drop all dimensions down to a pure 0D scalar.
    #         # This completely avoids the "requested shape has 0" C++ compiler crash!
    #         labels_tensor = tf.squeeze(labels_tensor)
    #         labels_tensor = tf.ensure_shape(labels_tensor, [])
    #     else:
    #         # Multi-class parallel datasets maintain their 1D class vectors
    #         labels_tensor = tf.reshape(labels_tensor, [n_classes])
    #         labels_tensor = tf.ensure_shape(labels_tensor, [n_classes])

    #     return tf.cast(features_tensor, tf.float32), tf.cast(labels_tensor, tf.float32)



    # train = train.unbatch()
    # test = test.unbatch()
    # val = val.unbatch()

    print("before map_train ")
    print(train.element_spec)
    print(val.element_spec)

    # train = train.batch(1)
    # test = test.batch(1)
    # val = val.batch(1)



    # --- FIX AUTOGRAPH WARNINGS: Use a named wrapper instead of a lambda ---
    def map_train(features, labels):
        return _preprocess(features, labels, n_classes=config['output_dim'])
        
    def map_test(features, labels):
        return _preprocess(features, labels, n_classes=config['output_dim'])
        
    def map_val(features, labels):
        return _preprocess(features, labels, n_classes=config['output_dim'])

    # Apply the clean map functions
    train = train.map(map_train)
    test = test.map(map_test)
    val = val.map(map_val)

    # print("--- VERIFIED UNBATCHED SPEC ---")
    # print(train.element_spec)

    # for a,b in train.batch(1).take(1):
    #     print()
    #     print("Sanity check")
    #     print(a[0].numpy().shape,b[0].numpy())

    config['train'] = train
    config['test'] = test
    config['val'] = val
    
    return config

def get_test_set(train_path):
    if not "train" in train_path:
        print("Please provide the path to a train dataset as the train_path parameter")
        raise ValueError
    config = {}
    prefix = train_path.split('_train')[0]
    config['name'] = os.path.basename(prefix)
    config['prefix'] = prefix
    config['train_path'] = train_path
    test = tf.data.Dataset.load(train_path.replace('train', 'test'), compression="GZIP")
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
                # 2. Create the initializer that reads your file line by line
        keys = list(vocabulary.keys())
        values = list(vocabulary.values())

        if "[UNK]" in keys:
            print("Found [UNK]")
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=keys,
                values=values,
                key_dtype=tf.string,
                value_dtype=tf.int64
            )
    
            table = tf.lookup.StaticVocabularyTable(
                initializer, 
                num_oov_buckets=1
            )
        else:
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=keys, 
                values=values, 
                key_dtype=tf.string, 
                value_dtype=tf.int64
            )

            # Use StaticHashTable instead of StaticVocabularyTable
            table = tf.lookup.StaticHashTable(initializer, default_value=-1)
            
        config['table'] = table

    except FileNotFoundError as e:
        print('Assuming binary')
        config['output_dim'] = 1 #assume binary classification
        config['binary'] = True

    
    def _set_dimensions(train):
        original_batches = tf.data.experimental.cardinality(train).numpy()
        for batch in train.take(1):
            features, labels = batch
            batch_size, input_dim = features.shape

            total_samples = original_batches * batch_size
            steps_per_epoch = int(total_samples // batch_size)
            print(f"set_dimensions {steps_per_epoch=}")
        
            is_categorical = (
                isinstance(labels, pd.Categorical) or 
                pd.api.types.is_categorical_dtype(labels) or 
                np.issubdtype(np.asarray(labels).dtype, np.object_) or 
                np.issubdtype(np.asarray(labels).dtype, np.str_)
            )
            break
        return batch_size, input_dim, is_categorical, total_samples

    batch_size, input_dim, is_categorical, total_samples = _set_dimensions(test)
    print(f"{batch_size=}, {input_dim=}, {is_categorical=}, {total_samples=}")
    config['source_batch_size'] = batch_size
    config['input_dim'] = input_dim
    config['total_samples'] = total_samples

    test = test.unbatch()

    for a,b in test.batch(1).take(1):
        print()
        print("Sanity check")
        print(a[0].numpy().shape,b[0].numpy())
    
    def _preprocess(features, labels, n_classes=config['output_dim'], config=config):
        labels_tensor = tf.convert_to_tensor(labels)
        labels_tensor = tf.cast(labels_tensor, tf.float32)

        # Use Python logic driven by your config instead of tf.cond.
        # This keeps the TensorShape fully known and static for Keras!
        if n_classes == 1:
            # Explicitly force a 2D matrix shape of (None, 1)
            # This handles both batched and unbatched binary input safely
            labels_tensor = tf.reshape(labels_tensor, [-1, 1])
            labels_tensor = tf.ensure_shape(labels_tensor, [None, 1])
        else:
            # For one-hot datasets, lock the final dimension to n_classes
            labels_tensor = tf.reshape(labels_tensor, [-1, n_classes])
            labels_tensor = tf.ensure_shape(labels_tensor, [None, n_classes])

        return features, labels_tensor




    print(f"in preprocess: {config['output_dim']=}")

    config['test'] = test
    
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
