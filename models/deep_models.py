"""
Deep Learning Models Module
Provides TensorFlow/Keras implementations for 1D-CNN, LSTM, and CNN-LSTM architectures
used for EEG time-series and audio spectrogram feature modeling.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
from utils.logger import logger

class DeepModelSuite:
    """
    Suite for building, training, and evaluating TensorFlow Deep Learning models.
    """
    @staticmethod
    def build_cnn_1d(input_shape, num_classes=1):
        """Builds a 1D Convolutional Neural Network."""
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=(input_shape[0],)),
            Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    @staticmethod
    def build_lstm(input_shape, num_classes=1):
        """Builds an LSTM Recurrent Neural Network."""
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=(input_shape[0],)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    @staticmethod
    def build_cnn_lstm(input_shape, num_classes=1):
        """Builds a hybrid CNN-LSTM network."""
        model = Sequential([
            Reshape((input_shape[0], 1), input_shape=(input_shape[0],)),
            Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(32),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    @classmethod
    def train_and_evaluate_deep_models(cls, X_train: np.ndarray, y_train: np.ndarray, 
                                       X_val: np.ndarray, y_val: np.ndarray, 
                                       epochs: int = 10, batch_size: int = 8):
        """
        Trains CNN, LSTM, and CNN-LSTM models and compares performance.
        
        Args:
            X_train, y_train: Training split
            X_val, y_val: Validation split
            epochs (int): Max epochs
            batch_size (int): Mini-batch size
            
        Returns:
            dict: Model training histories and evaluation metrics.
            tuple: (best_dl_name, best_dl_model)
        """
        input_shape = (X_train.shape[1],)
        dl_architectures = {
            'CNN': cls.build_cnn_1d(input_shape),
            'LSTM': cls.build_lstm(input_shape),
            'CNN-LSTM': cls.build_cnn_lstm(input_shape)
        }
        
        results = {}
        best_score = -1.0
        best_dl_name = ""
        best_dl_model = None
        
        early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)

        for name, model in dl_architectures.items():
            logger.info(f"Training Deep Learning model architecture: {name}")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            val_probs = model.predict(X_val, verbose=0).flatten()
            val_preds = (val_probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_val, val_preds)
            f1 = f1_score(y_val, val_preds, zero_division=0)
            auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
            
            results[name] = {
                'model': model,
                'history': history.history,
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc,
                'val_probs': val_probs
            }
            
            logger.info(f"[{name}] Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Val AUC: {auc:.4f}")
            
            if auc > best_score:
                best_score = auc
                best_dl_name = name
                best_dl_model = model
                
        return results, (best_dl_name, best_dl_model)
