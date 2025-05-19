from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Reshape, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
import tensorflow as tf


def build_hybrid_model():
    # 1. Image Input Branch
    img_input = Input(shape=(50, 50, 3), name='img_input')
    x1 = Conv2D(32, 3, padding='same')(img_input)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D()(x1)

    x1 = Conv2D(64, 3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D()(x1)

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)

    # 2. Edge Map Branch
    edge_input = Input(shape=(50, 50, 1), name='edge_input')
    x2 = Conv2D(16, 3, padding='same')(edge_input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D()(x2)

    x2 = Flatten()(x2)
    x2 = Dropout(0.3)(x2)

    # 3. Histogram Input
    hist_input = Input(shape=(256,), name='hist_input')
    x3 = Dense(128)(hist_input)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.3)(x3)

    # 4. Concatenate all features
    combined = Concatenate()([x1, x2, x3])

    # 5. Self-attention block
    attn_input = Reshape((1, -1))(combined)
    attn = LayerNormalization()(attn_input)
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(attn, attn)
    attn = GlobalAveragePooling1D()(attn)

    # 6. Classification Head
    x = Dense(256)(attn)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    output = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=[img_input, edge_input, hist_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
