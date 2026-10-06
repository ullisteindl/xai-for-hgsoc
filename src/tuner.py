from tensorboard import notebook
import tensorflow as tf
from deel.lip.layers import SpectralDense, FrobeniusDense
from deel.lip.activations import GroupSort2
from deel.lip.model import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

from tensorflow.keras import layers, Model

import random
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.metrics import binary_accuracy
import sklearn.metrics
import matplotlib.pyplot as plt
import itertools
import io
import keras_tuner as kt
from deel.lip.losses import MulticlassHKR, HKR, KR, HingeMargin
from deel.lip.metrics import CategoricalProvableAvgRobustness

from tensorflow.keras.optimizers import Adam
import json
import math
import os
import gc
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


class MyHyperModel(kt.HyperModel):

    def __init__(self, input_data=None, **kwargs):
        super().__init__(**kwargs)
        # self.active_trial_id = 'default'
        if input_data:
            self.train = input_data['train']
            self.test = input_data['test']
            self.val = input_data['val']
            self.model_name = input_data['name']
            self.prefix = input_data['prefix']
            self.output_dim = input_data['output_dim']
            self.binary = input_data['binary']
            self.input_dim = input_data['input_dim']
            self.source_batch_size = input_data['source_batch_size']
            self.total_samples = input_data['total_samples']
            self.spe = int(math.ceil(self.total_samples / self.source_batch_size ))
            
    def on_trial_end(self, trial):  
        super(MyCustomTuner, self).on_trial_end(trial)
        
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"--- Cleared GPU memory after Trial {trial.trial_id} ---")

    def on_trial_begin(self, trial):
        """Runs automatically right before a trial starts."""
        super(MyCustomTuner, self).on_trial_begin(trial)
        
        try:
            # Get memory details for GPU 0 (returns dict with 'current' and 'peak' in bytes)
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            
            # Convert bytes to Megabytes (MB) for easier reading
            current_vram = gpu_info['current'] / (1024 ** 2)
            peak_vram = gpu_info['peak'] / (1024 ** 2)
            
            print(f"\n[Trial {trial.trial_id} Start] GPU Memory Allocated: {current_vram:.2f} MB (Peak: {peak_vram:.2f} MB)")
            sys.stdout.flush() 
        
        except ValueError:
            print(f"\n[Trial {trial.trial_id} Start] Could not read GPU memory info (Are you running on CPU?)")


    def initialize_callbacks(self, total_epochs=20, save_x_times=10, steps_per_epoch=None, log_path = None, model_name=""):
        if self.spe:
            steps_per_epoch = int(self.spe)
            print(f"{self.spe=}")
        if self.model_name:
            log_path = f'logs/hptune/{self.model_name}'
        
        save_every_x_epochs = int(total_epochs // save_x_times)
        print(f"{save_every_x_epochs=} = {total_epochs=} // {save_x_times=}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = log_path + timestamp

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/{timestamp}/model_at_epoch_{{epoch:02d}}',
            verbose=1,
            save_freq=int(save_every_x_epochs * steps_per_epoch)  # Save every X epochs
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        file_writer_cm = tf.summary.create_file_writer(log_path + '/cm')

        if self.binary:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",      # Metric to monitor
                    patience=2,             # Number of epochs to wait before stopping
                    restore_best_weights=True # Rewinds model to the best epoch's weights
                )
        else:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                    monitor="val_weighted_f1",      # Metric to monitor
                    patience=2,             # Number of epochs to wait before stopping
                    restore_best_weights=True, # Rewinds model to the best epoch's weights
                    mode = 'max'
                )
            
        best_model_callback = keras.callbacks.ModelCheckpoint(
                filepath=f"models/{timestamp}/best_model.keras", # Path where model is saved
                monitor="val_weighted_f1",          # Metric to monitor
                save_best_only=True          # Only overwrite if performance improved
            )
        

        # cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
        return [checkpoint_callback, tensorboard_callback, early_stopping_callback, best_model_callback]

    def build(self, hp):

        large_architecture = layer_configs = [
            {
                "type": SpectralDense,
                "units": 4096, 
                "activation": GroupSort2(),
                "use_bias": True, 
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-1),
                "name": "spectral_dense_4096"
            },
            {
                "type": SpectralDense,
                "units": 1024, 
                "activation": GroupSort2(),
                "use_bias": True, 
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-2),
                "name": "spectral_dense_1024"
            },

            {
                "type": SpectralDense,
                "units": 256, 
                "activation": GroupSort2(),
                "use_bias": True, 
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-3),
                "name": "spectral_dense_256"
            },
            {
                "type": SpectralDense,
                "units": 256,
                "activation": GroupSort2(),
                "use_bias": True,
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-4),
                "name": "spectral_dense_256_2"
            }
        ]
        
        small_architecture = [
                                    {
                "type": SpectralDense,
                "units": 64, 
                "activation": GroupSort2(),
                "use_bias": True, 
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-5),
                "name": "spectral_dense_64"
            },

                        {
                "type": SpectralDense,
                "units": 32, 
                "activation": GroupSort2(),
                "use_bias": True, 
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-6),
                "name": "spectral_dense_32"
            },
            {
                "type": SpectralDense,
                "units": 16,
                "activation": GroupSort2(),
                "use_bias": True,
                "kernel_initializer": tf.keras.initializers.Orthogonal(seed=SEED-7),
                "name": "spectral_dense_16"
            }
        ]
        
        if "treated" in self.model_name:
            print('setting special architecture for only treated model')
            small_architecture = [
                                        {
                    "type": SpectralDense,
                    "units": 256, 
                    "activation": GroupSort2(),
                    "use_bias": True, 
                    "name": "spectral_dense_256"
                },
                                        {
                    "type": SpectralDense,
                    "units": 64, 
                    "activation": GroupSort2(),
                    "use_bias": True, 
                    "name": "spectral_dense_64"
                },
                            {
                    "type": SpectralDense,
                    "units": 32, 
                    "activation": GroupSort2(),
                    "use_bias": True, 
                    "name": "spectral_dense_32"
                }
            ]

        alpha = hp.Int("alpha", 5, 100, sampling="log", default=10)
        optimizer = keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        min_margin = hp.Float("min_margin", 0.1, 1, sampling="log", default=0.25)

        if self.binary:
            relevant_metrics = [
                'accuracy', 
                tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                # tf.keras.metrics.F1Score(average='micro', name='f1', threshold=None),
                # tf.keras.metrics.F1Score(threshold=0.5, name='f1'),
                KR(),  
                HingeMargin(min_margin=min_margin),
                # CategoricalProvableAvgRobustness(name="Robust"),
                # CategoricalProvableAvgRobustness(negative_robustness=True, name="margin")
                HKR_binary_accuracy,
                HKR_precision,
                HKR_recall,
                HKR_f1_score
            ]
        else:
            relevant_metrics = [
                'categorical_accuracy', 
                tf.keras.metrics.AUC(from_logits=True, multi_label=True, num_thresholds=200, name="multiclass_auc"),
                tf.keras.metrics.F1Score(average='weighted', name='weighted_f1', threshold=None),
                MulticlassHKR(),
                CategoricalProvableAvgRobustness(name="Robust"),
                CategoricalProvableAvgRobustness(negative_robustness=True, name="margin")]
        
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        if self.input_dim > 1000:
            print('setting model size')
            use_layers = hp.Choice('model_size', values=[(len(large_architecture) - 1) , len(large_architecture)])
            layer_configs = large_architecture[:use_layers]
            print(f"{len(layer_configs)=}")
        else:
            use_layers = 0
            layer_configs = small_architecture
        
        for config in layer_configs:
            cfg = config.copy()
            layer_class = cfg.pop("type")
        
            keras_layer = layer_class(**cfg)
            x = keras_layer(x)
    
        outputs = FrobeniusDense(self.output_dim, activation=None, use_bias=False,
                               kernel_initializer=tf.keras.initializers.Orthogonal(seed=SEED+2))(x)
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        optimizer = keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        alpha = hp.Int("alpha", 5, 15, sampling="log", default=10)
        min_margin = hp.Float("min_margin", 0.1, 1, sampling="log", default=0.25)
        if self.binary:
            loss = HKR(alpha=alpha, min_margin=min_margin)
        else:
            loss = MulticlassHKR(alpha=alpha, min_margin=min_margin)

            
        model.compile(optimizer=optimizer, loss=loss, metrics=relevant_metrics)

        print()
        print("Set up model for Hyperparameter tuning")
        print(model.summary())
        return model

    def fit(self, hp, model, x, y=None, *args, **kwargs):
        available_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        chosen_batch_size = hp.Choice('batch_size', values=available_sizes)

        print(f"{chosen_batch_size=}")
        print(f"{self.binary=}")
        actual_integer_value = hp.get('batch_size')
        print(f"The actual integer value chosen for this trial is: {actual_integer_value=}")

        self.spe = self.total_samples / chosen_batch_size

        # 1. Cleanly batch the primary training dataset (x)
        # Since x is an explicit argument, we handle it directly here

        # print("before batching in tuner")
        # print(x.element_spec)
        batched_train_ds = x.batch(chosen_batch_size).prefetch(tf.data.AUTOTUNE)
        print("after batching in tuner")
        print(batched_train_ds.element_spec)

        if "validation_data" in kwargs and kwargs["validation_data"] is not None:
            raw_val_ds = kwargs.pop("validation_data")
            
            # 1. FIX: Add drop_remainder=True to prevent the 509 remainder crash
            kwargs["validation_data"] = raw_val_ds.batch(chosen_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
            print("after batching validation data")
            # # 2. FIX: Print the element_spec of the NEW batched object from kwargs
            print(kwargs["validation_data"].element_spec)

        # # 2. Cleanly batch validation data if it exists in kwargs
        # if "validation_data" in kwargs and kwargs["validation_data"] is not None:
        #     raw_val_ds = kwargs.pop("validation_data")
        #     kwargs["validation_data"] = raw_val_ds.batch(chosen_batch_size).prefetch(tf.data.AUTOTUNE)
        #     print("after batching validation data")
        #     print(raw_val_ds.element_spec)
  
        # 3. Handle callbacks safely
        tuner_callbacks = kwargs.pop("callbacks", []) 
        my_callbacks = self.initialize_callbacks()
        
        # 4. Inject updated training parameters
        kwargs["batch_size"] = chosen_batch_size
        kwargs["steps_per_epoch"] = self.spe

        # 5. Pass batched_train_ds explicitly as the first argument to model.fit
        return model.fit(
            batched_train_ds,
            y=y,
            *args,
            callbacks=tuner_callbacks + my_callbacks,
            **kwargs
        )


class NeuralNetHyperModel(kt.HyperModel):
    def __init__(self, input_data=None, **kwargs):
        super().__init__(**kwargs)
        # self.active_trial_id = 'default'
        if input_data:
            self.train = input_data['train']
            self.test = input_data['test']
            self.val = input_data['val']
            self.model_name = input_data['name']
            self.prefix = input_data['prefix']
            self.output_dim = input_data['output_dim']
            self.binary = input_data['binary']
            self.input_dim = input_data['input_dim']
            self.source_batch_size = input_data['source_batch_size']
            self.total_samples = input_data['total_samples']
            self.spe = int(math.ceil(self.total_samples / self.source_batch_size ))
            

    def on_trial_end(self, trial):  
        super(MyCustomTuner, self).on_trial_end(trial)
        
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"--- Cleared GPU memory after Trial {trial.trial_id} ---")

    def on_trial_begin(self, trial):
        """Runs automatically right before a trial starts."""
        super(MyCustomTuner, self).on_trial_begin(trial)
        
        try:
            # Get memory details for GPU 0 (returns dict with 'current' and 'peak' in bytes)
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            
            # Convert bytes to Megabytes (MB) for easier reading
            current_vram = gpu_info['current'] / (1024 ** 2)
            peak_vram = gpu_info['peak'] / (1024 ** 2)
            
            print(f"\n[Trial {trial.trial_id} Start] GPU Memory Allocated: {current_vram:.2f} MB (Peak: {peak_vram:.2f} MB)")
            sys.stdout.flush() 
        
        except ValueError:
            print(f"\n[Trial {trial.trial_id} Start] Could not read GPU memory info (Are you running on CPU?)")



    def initialize_callbacks(self, total_epochs=20, save_x_times=10, steps_per_epoch=None, log_path = None):
        if self.spe:
            steps_per_epoch = int(self.spe)
            print(f"{self.spe=}")
        if self.model_name:
            log_path = f'logs/hptune/{self.model_name}'
        
        save_every_x_epochs = int(total_epochs // save_x_times)
        print(f"{save_every_x_epochs=} = {total_epochs=} // {save_x_times=}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = log_path + timestamp

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/{timestamp}/{self.model_name}_at_epoch_{{epoch:02d}}',
            verbose=1,
            save_freq=int(save_every_x_epochs * steps_per_epoch)  # Save every X epochs
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        file_writer_cm = tf.summary.create_file_writer(log_path + '/cm')

        if self.binary:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",      # Metric to monitor
                    patience=10,             # Number of epochs to wait before stopping
                    restore_best_weights=True # Rewinds model to the best epoch's weights
                )
        else:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                    monitor="val_weighted_f1",      # Metric to monitor
                    patience=2,             # Number of epochs to wait before stopping
                    restore_best_weights=True, # Rewinds model to the best epoch's weights
                    mode = 'max'
                )
            
        best_model_callback = keras.callbacks.ModelCheckpoint(
                filepath=f"models/{timestamp}/best_model.keras", # Path where model is saved
                monitor="val_accuracy",          # Metric to monitor
                save_best_only=True          # Only overwrite if performance improved
            )
        # cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

        return [checkpoint_callback, tensorboard_callback, early_stopping_callback, best_model_callback]

    def build(self, hp):
    # Architectural configurations using standard Keras Dense layers
        large_architecture = [
            { "units": 4096, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(seed=SEED), "name": "dense_4096" },
            { "units": 1024, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(seed=SEED+1), "name": "dense_1024" },
            { "units": 256, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(seed=SEED+2), "name": "dense_256" },
            { "units": 256, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(seed=SEED+3), "name": "dense_256_2" }
        ]
        
        small_architecture = [
            { "units": 64, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(), "name": "dense_64" },
            { "units": 32, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(), "name": "dense_32" },
            { "units": 16, "activation": "relu", "use_bias": True, "kernel_initializer": tf.keras.initializers.HeNormal(), "name": "dense_16" }
        ]


        
        # Hyperparameters
        optimizer = tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        
        # Metrics and Losses for standard Dense classification
        if self.binary:
            relevant_metrics = [
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'), 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(average='micro', name='f1', threshold=None)
            ]
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            relevant_metrics = [
                'categorical_accuracy', 
                tf.keras.metrics.AUC(from_logits=True, multi_label=True, num_thresholds=200, name="multiclass_auc"),
                tf.keras.metrics.F1Score(average='weighted', name='weighted_f1', threshold=None)
            ]
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        # Build Model Graph
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        
        if self.input_dim > 1000:
            print('Setting model size')
            # Fixed Keras Tuner tracking logic using clean integers
            num_layers = hp.Choice('model_size', values=[3, 4]) 
            layer_configs = large_architecture[:num_layers]
            print(f"use_layers={num_layers}")
            print(f"len(layer_configs)={len(layer_configs)}")
        else:
            layer_configs = small_architecture
        
        # Dynamically unpack and build standard Dense layers
        for config in layer_configs:
            cfg = config.copy()
            keras_layer = Dense(**cfg)
            x = keras_layer(x)
        
        # Output layer using standard Dense
        # activation=None is used because from_logits=True is set in the losses above
        outputs = Dense(
            self.output_dim, 
            activation=None, 
            use_bias=True, 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED+2),
            name="output_layer"
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f"{len(layer_configs)}h_{self.model_name}")
        model.compile(optimizer=optimizer, loss=loss, metrics=relevant_metrics)
        
        print()
        print("Set up Dense model for Hyperparameter tuning")
        print(model.summary())
        return model

    def fit(self, hp, model, x, y=None, *args, **kwargs):
        available_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        chosen_batch_size = hp.Choice('batch_size', values=available_sizes)

        print(f"{chosen_batch_size=}")
        actual_integer_value = hp.get('batch_size')
        print(f"The actual integer value chosen for this trial is: {actual_integer_value=}")

        self.spe = self.total_samples / chosen_batch_size

        # 1. Cleanly batch the primary training dataset (x)
        # Since x is an explicit argument, we handle it directly here
        batched_train_ds = x.batch(chosen_batch_size).prefetch(tf.data.AUTOTUNE)

        # 2. Cleanly batch validation data if it exists in kwargs
        if "validation_data" in kwargs and kwargs["validation_data"] is not None:
            raw_val_ds = kwargs.pop("validation_data")
            kwargs["validation_data"] = raw_val_ds.batch(chosen_batch_size).prefetch(tf.data.AUTOTUNE)
  
        # 3. Handle callbacks safely
        tuner_callbacks = kwargs.pop("callbacks", []) 
        my_callbacks = self.initialize_callbacks()
        
        # 4. Inject updated training parameters
        kwargs["batch_size"] = chosen_batch_size
        kwargs["steps_per_epoch"] = self.spe

        # 5. Pass batched_train_ds explicitly as the first argument to model.fit
        return model.fit(
            batched_train_ds,
            y=y,
            *args,
            callbacks=tuner_callbacks + my_callbacks,
            **kwargs
        )

class MemoryTrackingHyperband(kt.Hyperband):
    
    def on_trial_begin(self, trial):
        """Runs automatically right before a trial starts."""
        super().on_trial_begin(trial)
        
        try:
            # Fetch TF memory stats for GPU 0
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            current_vram = gpu_info['current'] / (1024 ** 2)
            peak_vram = gpu_info['peak'] / (1024 ** 2)
            
            print(f"\n[Trial {trial.trial_id} Start] GPU Memory Allocated: {current_vram:.2f} MB (Peak: {peak_vram:.2f} MB)", flush=True)
        except ValueError:
            print(f"\n[Trial {trial.trial_id} Start] Could not read GPU memory info.", flush=True)
            
        sys.stdout.flush()

    def on_trial_end(self, trial):
        """Runs automatically right after a trial finishes."""
        super().on_trial_end(trial)

        # Force clear GPU memory session and trigger Python garbage collection
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"--- Cleared GPU memory after Trial {trial.trial_id} ---\n", flush=True)
        sys.stdout.flush()
 

@keras.saving.register_keras_serializable()
def HKR_binary_accuracy(y_true, y_pred):
    S_true = tf.dtypes.cast(tf.greater_equal(y_true[:, 0], 0), dtype=tf.float32)
    S_pred = tf.dtypes.cast(tf.greater_equal(y_pred[:, 0], 0), dtype=tf.float32)
    return binary_accuracy(S_true, S_pred)


@keras.saving.register_keras_serializable()
def HKR_precision(y_true, y_pred):
    S_true = tf.dtypes.cast(tf.greater_equal(y_true[:, 0], 0), dtype=tf.float32)
    S_pred = tf.dtypes.cast(tf.greater_equal(y_pred[:, 0], 0), dtype=tf.float32)
    
    true_positives = tf.reduce_sum(S_true * S_pred)
    predicted_positives = tf.reduce_sum(S_pred)
    
    result = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    # --- THE FIX ---
    return tf.reshape(result, []) 

@keras.saving.register_keras_serializable()
def HKR_recall(y_true, y_pred):
    S_true = tf.dtypes.cast(tf.greater_equal(y_true[:, 0], 0), dtype=tf.float32)
    S_pred = tf.dtypes.cast(tf.greater_equal(y_pred[:, 0], 0), dtype=tf.float32)
    
    true_positives = tf.reduce_sum(S_true * S_pred)
    actual_positives = tf.reduce_sum(S_true)
    
    result = true_positives / (actual_positives + tf.keras.backend.epsilon())
    # --- THE FIX ---
    return tf.reshape(result, []) 

@keras.saving.register_keras_serializable()
def HKR_f1_score(y_true, y_pred):
    p = HKR_precision(y_true, y_pred)
    r = HKR_recall(y_true, y_pred)
    
    result = 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    return tf.reshape(result, []) 

