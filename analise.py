import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Carrega o modelo treinado
modelo = tf.keras.models.load_model("modelo_extintores_iso14001.h5")

# Dicionário das classes (confirme a ordem exata gerada pelo ImageDataGenerator)
classes = {
    0: "conforme",
    1: "fora_posicao",
    2: "sem_placa",
    3: "obstruido",
    4: "ausente"
}

def analisar_imagem(caminho):
    img = image.load_img(caminho, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)[0]
    classe_idx = np.argmax(pred)
    confianca = pred[classe_idx] * 100

    print(f"🔍 Resultado: {classes[classe_idx]} ({confianca:.2f}% de confiança)")

# Exemplo de uso
analisar_imagem("teste_extintor.jpg")
