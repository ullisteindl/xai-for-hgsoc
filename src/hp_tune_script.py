import os
import sys
import argparse
import datetime
import json

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow import keras

# deel-lip imports (not used unless requested)
try:
    from deel.lip.layers import SpectralDense, FrobeniusDense
    from deel.lip.activations import GroupSort2
    from deel.lip.losses import MulticlassHKR, HKR, KR, HingeMargin
    from deel.lip.metrics import CategoricalProvableAvgRobustness
except Exception:
    SpectralDense = None

from prep_data_get_dimensions import prep_data_get_dimensions
from tuner import MemoryTrackingHyperband
import math


def initialize_callbacks(log_path="logs/hptune_nn/", model_name="", track=None, patience=5):
    if model_name:
        log_path = f"{log_path}/{model_name}/"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_path, timestamp)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models', model_name, 'model_at_epoch_{epoch:02d}'),
        verbose=1,
        save_freq='epoch'
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if track:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=track,
            patience=patience,
            restore_best_weights=True,
            mode='max'
        )
        best_model_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('models', f'{model_name}_best.keras'),
            monitor=track,
            save_best_only=True
        )
        return [checkpoint_callback, tensorboard_callback, early_stopping_callback, best_model_callback]
    else:
        return [checkpoint_callback, tensorboard_callback]


class DenseHyperModel(kt.HyperModel):
    def __init__(self, input_data, architecture='baseline', hidden_layers=None):
        self.input_data = input_data
        self.architecture = architecture
        self.hidden_layers = hidden_layers or []
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

    def build(self, hp):
        input_dim = self.input_data['input_dim']
        outputs_dim = self.input_data['output_dim']

        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)

        # Ensure input shape is correct whether input_dim is int or tuple
        if isinstance(input_dim, int):
            input_shape = (input_dim,)
        else:
            input_shape = tuple(input_dim)

        inputs = Input(shape=input_shape)
        x = inputs

        # fixed user-provided hidden layers
        for i, units in enumerate(self.hidden_layers):
            units = int(units)
            if self.architecture == 'deel' and SpectralDense is not None:
                print('setting up hidden layers')
                x = SpectralDense(units, activation=GroupSort2, name=f'spectral_{i}', use_bias=True)(x)
                if i == len(self.hidden_layers)-1:
                    outputs = FrobeniusDense(output_dim, activation=None, name='output_layer', use_bias=False,
                               kernel_initializer=tf.keras.initializers.Orthogonal(seed=i))(x)
            else:
                initializer = tf.keras.initializers.HeNormal(seed=42 + i)
                x = Dense(units, activation='relu', kernel_initializer=initializer, name=f'dense_{i}')(x)
                if i == len(self.hidden_layers)-1:
                    outputs = Dense(outputs_dim, activation=None, name='output_layer', use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal(seed=43 + 1))(x)

        model = Model(inputs=inputs, outputs=outputs)

        # choose optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # choose loss & metrics based on binary vs multiclass
        if self.architecture == 'baseline':
            if outputs_dim <= 1:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            alpha = hp.Int("alpha", 5, 100, sampling="log", default=10)
            min_margin = hp.Float("min_margin", 0.1, 1, sampling="log", default=0.25)
            if outputs_dim <= 1:
                loss = HKR(alpha=alpha, min_margin=min_margin)
            else:
                loss = MulticlassHKR(alpha=alpha, min_margin=min_margin)

        if outputs_dim <= 1:
            relevant_metrics = [
                'accuracy', 
                tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                KR(),
                HingeMargin(min_margin=min_margin),
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

        model.compile(optimizer=optimizer, loss=loss, metrics=relevant_metrics)
        return model

    def fit(self, hp, model, x, y=None, *args, **kwargs):
        available_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        chosen_batch_size = hp.Choice('batch_size', values=available_sizes)

        print(f"{chosen_batch_size=}")
        actual_integer_value = hp.get('batch_size')
        self.spe = self.total_samples / chosen_batch_size

        # Batch train
        batched_train_ds = x.batch(chosen_batch_size).prefetch(tf.data.AUTOTUNE)

        # Batch val
        if "validation_data" in kwargs and kwargs["validation_data"] is not None:
            raw_val_ds = kwargs.pop("validation_data")
            
            kwargs["validation_data"] = raw_val_ds.batch(chosen_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
  
        # re-initiate callbacks
        tuner_callbacks = kwargs.pop("callbacks", []) 
        my_callbacks = initialize_callbacks()
        
        # make sure training params are accessible
        kwargs["batch_size"] = chosen_batch_size
        kwargs["steps_per_epoch"] = self.spe

        # Pass batched_train_ds to model.fit
        return model.fit(
            batched_train_ds,
            y=y,
            *args,
            callbacks=tuner_callbacks + my_callbacks,
            **kwargs
        )


def parse_hidden_layers(s):
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def choose_objective(data, hypermodel, args):
    binary = bool(data.get('binary', False))

    # Match the notebook behavior exactly for the Deel/LIP hypermodel from src.tuner.
    if args.architecture == 'deel':
        if binary:
            return kt.Objective('val_HKR_binary_accuracy', direction='max')
        return kt.Objective('val_weighted_f1', direction='max')
    else:
    # Standard dense MLP fallback keeps the usual Keras metric names.
        if binary:
            return kt.Objective('val_accuracy', direction='max')
        return kt.Objective('val_weighted_f1', direction='max')


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning and training script adapted from notebooks')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset folder (same as used in notebooks)')
    parser.add_argument('--architecture', choices=['baseline', 'deel'], default='baseline')
    parser.add_argument('--hidden_layers', type=str, default='64,32,16', help='Comma-separated hidden layer sizes, e.g. 256,256,128,64')
    parser.add_argument('--trials', type=int, default=10, help='Number of top trials to save / tune budget (used for reporting)')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--project_name', type=str, default='hp_tuning_script')
    args = parser.parse_args()

    hidden_layers = parse_hidden_layers(args.hidden_layers)

    # Prepare data (mirrors notebook usage)
    if args.architecture == 'baseline':
        data = prep_data_get_dimensions(args.data, for_MLP=True)
    else:
        data = prep_data_get_dimensions(args.data, for_MLP=False)
        

    # Choose HyperModel implementation from src.tuner
    # if args.architecture == 'baseline':
    #     hypermodel = NeuralNetHyperModel(input_data=data)
    # else:
    #     hypermodel = MyHyperModel(input_data=data)

    hypermodel = DenseHyperModel(input_data=data, architecture=args.architecture, hidden_layers=hidden_layers)

    # Objective selection mirrors the notebook's tuner.py behavior.
    objective = choose_objective(data, hypermodel, args)
    print(f"Tuning objective: {objective.name} ({objective.direction})")

    # Use MemoryTrackingHyperband from src.tuner for parity with notebooks
    tuner = MemoryTrackingHyperband(
        hypermodel=hypermodel,
        objective=objective,
        max_epochs=args.max_epochs,
        factor=2,
        hyperband_iterations=2,
        directory=args.project_name,
        project_name=data.get('name', args.project_name),
        overwrite=True
    )

    # Prepare datasets (pass raw datasets; custom HyperModel.fit handles batching)
    train = data['train']
    val = data['val']

    tuner.search(x=train, validation_data=val, epochs=min(5, args.max_epochs))

    best_trials = tuner.oracle.get_best_trials(num_trials=min(args.trials, 20))
    top_data = []
    for t in best_trials:
        d = t.hyperparameters.values.copy()
        d['score'] = t.score
        d['trial_id'] = t.trial_id
        top_data.append(d)

    df_top = pd.DataFrame(top_data)
    os.makedirs(tuner.directory, exist_ok=True)
    top_csv = os.path.join(tuner.directory, f"top_{len(df_top)}_{data.get('name','run')}.csv")
    df_top.to_csv(top_csv, index=False)
    print(f"Saved top trials to {top_csv}")

    # Train best hyperparameters (use the best hyperparameters returned by tuner)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    if best_hps:
        best_hp = best_hps[0]
        print('Best hyperparameters:', best_hp.values)
        # Rebuild model with best hyperparameters
        model = hypermodel.build(best_hp)

        # finalize datasets for training: batch according to best hyperparameter batch_size if available
        batch_size = int(best_hp.values.get('batch_size', 64)) if 'batch_size' in best_hp.values else 64
        train_b = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_b = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # compute steps_per_epoch for callbacks that use it
        total_samples = data.get('total_samples', None)
        steps_per_epoch = None
        if total_samples:
            steps_per_epoch = int(math.ceil(total_samples / batch_size))

        # Use hypermodel's callback initializer (from src.tuner) for parity with notebooks
        try:
            callbacks = hypermodel.initialize_callbacks(total_epochs=args.max_epochs, steps_per_epoch=steps_per_epoch, log_path='logs/tuned_fit/')
        except TypeError:
            # fallback if signature differs
            callbacks = hypermodel.initialize_callbacks()

        model.fit(train_b, epochs=args.max_epochs, validation_data=val_b, callbacks=callbacks, verbose=1)
        # Save final model
        final_model_path = os.path.join('models', f"{data.get('name','model')}_final.keras")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
    else:
        print('No best hyperparameters found by tuner.')


if __name__ == '__main__':
    main()
