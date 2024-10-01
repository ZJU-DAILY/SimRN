import tensorflow as tf

class Date2VecConvert:
    def __init__(self, model_path="./d2v_model/d2v_98291_17.169918439404636"):
        self.model = tf.saved_model.load(model_path)
    
    def __call__(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)
        return self.model.encode(x)[0].numpy()

class Date2Vec(tf.keras.Model):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()
        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = tf.keras.layers.Dense(k1, activation='linear')
        self.fc2 = tf.keras.layers.Dense(k2, activation='linear')
        self.d2 = tf.keras.layers.Dropout(0.3)
        
        if act == 'sin':
            self.activation = tf.math.sin
        else:
            self.activation = tf.math.cos

        self.fc3 = tf.keras.layers.Dense(k, activation='linear')
        self.d3 = tf.keras.layers.Dropout(0.3)
        
        self.fc4 = tf.keras.layers.Dense(6, activation='linear')
        self.fc5 = tf.keras.layers.Dense(6, activation='linear')

    def call(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = tf.concat([out1, out2], axis=1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = tf.concat([out1, out2], axis=1)
        return out

if __name__ == "__main__":
    model = Date2Vec()
    inp = tf.random.normal([1, 6])

    out = model(inp)
    print(out)
    print(out.shape)
