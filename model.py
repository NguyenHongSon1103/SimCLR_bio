import tensorflow as tf
import tensorflow.keras.layers as L

class Model:
    def __init__(self, size_1=20217, size_2=17073):
        self.inp_A1 = L.Input(shape=(size_1))
        self.inp_A2 = L.Input(shape=(size_1))
        self.inp_B1 = L.Input(shape=(size_2))
        self.inp_B2 = L.Input(shape=(size_2))

        self.branch_A = tf.keras.Sequential(layers=[
            L.Dense(1024), L.ReLU(),
            L.Dense(512), L.ReLU(),
            L.Dense(256), L.ReLU()
        ], name='branch_1')
        self.branch_B = tf.keras.Sequential(layers=[
            L.Dense(1024), L.ReLU(),
            L.Dense(512), L.ReLU(),
            L.Dense(256), L.ReLU()
        ], name='branch_2')
        
        self.embedding = tf.keras.Sequential(layers=[
            L.Dense(256), L.ReLU(),
            L.Dense(128) 
        ], name='embedding')
        
        self.head = tf.keras.Sequential(layers=[
            L.ReLU(),
            L.Dense(128), L.ReLU(),
            L.Dense(64), L.ReLU()
        ], name='head')

    def make_model(self, is_training=True):
        out_A1 = self.branch_A(self.inp_A1)
        out_B1 = self.branch_B(self.inp_B1)
        out_A2 = self.branch_A(self.inp_A2)
        out_B2 = self.branch_B(self.inp_B2)
        x1 = L.Concatenate()([out_A1, out_B1])
        x2 = L.Concatenate()([out_A2, out_B2])
        emb_1 = self.embedding(x1)
        emb_2 = self.embedding(x2)
        out1 = self.head(emb_1) 
        out2 = self.head(emb_2)

        if is_training:
            # out = L.Concatenate(axis=0)([out1, out2])
            return tf.keras.Model(
                inputs=[self.inp_A1, self.inp_A2, self.inp_B1, self.inp_B2],
                outputs=[out1, out2])
        return tf.keras.Model(
                inputs=[self.inp_A1, self.inp_B1], 
                outputs = emb_1)

if __name__ == '__main__':
    import numpy as np
    A1 = np.random.random((5, 20217))
    A2 = np.random.random((5, 20217))
    B1 = np.random.random((5, 17073))
    B2 = np.random.random((5, 17073))
    training_model = Model().make_model(True)
    # training_model.summary()
    res = training_model([A1, A2, B1, B2])
    print(res[0].shape)
    print(res[1].shape)
    # infer_model = Model().make_model(False)
    # infer_model.summary()
