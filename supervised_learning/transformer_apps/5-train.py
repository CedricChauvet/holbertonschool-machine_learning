#!/usr/bin/env python3
"""
whole transformer model
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule personnalisé"""
    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)
def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Entraîne un modèle Transformer
    """
    # Préparation des données
    data = Dataset(batch_size, max_len)
    vocab_size = data.tokenizer_pt.vocab_size + 2
    
    # Création du modèle
    transformer = Transformer(
        N=N, 
        dm=dm, 
        h=h, 
        hidden=hidden,
        input_vocab=vocab_size,
        target_vocab=vocab_size,
        max_seq_input=max_len,
        max_seq_target=max_len
    )
    
    # Learning rate schedule
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    def masked_loss(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    # Compilation du modèle
    transformer.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    
    class TransformerDataset(tf.keras.utils.Sequence):
        def __init__(self, dataset, batch_size):
            self.dataset = list(dataset)  # Convertir en liste pour l'indexation
            self.batch_size = batch_size
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            inp, tar = self.dataset[idx]
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            # Création des masques
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            
            # Retourner les inputs sous forme de dictionnaire
            return {
                'inputs': inp,
                'target': tar_inp,
                'encoder_mask': enc_padding_mask,
                'look_ahead_mask': combined_mask,
                'decoder_mask': dec_padding_mask
            }, tar_real
    
    # Création des datasets
    train_dataset = TransformerDataset(data.data_train, batch_size)
    val_dataset = TransformerDataset(data.data_valid, batch_size)
    
    # Entraînement
    history = transformer.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'transformer_checkpoint.h5',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=2,
                restore_best_weights=True
            )
        ]
    )
    
    return transformer, history

# def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
#     """
#     to be filled
#     """
#     data = Dataset(batch_size, max_len)
#     vocab_size =  data.tokenizer_pt.vocab_size + 2
#     transformer = Transformer(N, dm, h, hidden, vocab_size, vocab_size,
#                               max_len, max_len)
#     print("transformer", type(transformer))

#     # 1. Compilation du modèle avec Adam et Sparse Categorical Crossentropy
#     transformer.compile(
#         optimizer='adam',  # Optimiseur Adam
#         loss='sparse_categorical_crossentropy',  # Perte Sparse Categorical Crossentropy
#         metrics=['accuracy']  # Métrique pour évaluer la performance
#     )

#     # 2. Entraînement du modèle
#     # X_train et y_train sont respectivement les données d'entrée et les labels
#     transformer.fit(
#         data.data_train,  # Données d'entraînement
#         # X_train,  # Données d'entraînement
#         # y_train,  # Labels correspondants
#         batch_size=32,  # Taille des lots d'entraînement
#         epochs=10,  # Nombre d'époques
#         validation_data=(data.data_valid)  # Validation sur un ensemble de validation (optionnel)
#     )