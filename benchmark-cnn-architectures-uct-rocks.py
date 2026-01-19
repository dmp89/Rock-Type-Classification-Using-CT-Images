# -*- coding: utf-8 -*-
"""
Script atualizado - Sem Data Leakage (GroupShuffleSplit)
Classe "Outros" tratada como imagens independentes
"""

# Importações
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152, DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, InceptionResNetV2, VGG16, VGG19, EfficientNetB0, EfficientNetB1,
    EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6,
    EfficientNetB7, MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    NASNetMobile, NASNetLarge, Xception
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# Função para calcular MPCA
def mean_per_class_accuracy(y_true, y_pred, num_classes):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    class_accuracies = []
    for i in range(num_classes):
        class_samples = tf.equal(y_true, i)
        class_samples = tf.cast(class_samples, tf.float32)
        total_samples = tf.reduce_sum(class_samples)
        if total_samples > 0:
            class_correct = tf.reduce_sum(class_samples * tf.cast(tf.equal(y_pred, i), tf.float32))
            class_accuracy = class_correct / total_samples
            class_accuracies.append(class_accuracy.numpy())
    return sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

# Callback para monitorar Overfitting e Underfitting
class OverUnderfitMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if epoch > 0:
            if val_loss > self.previous_val_loss and train_loss < self.previous_train_loss:
                print(f"\n⚠️ Overfitting detectado na época {epoch + 1}: val_loss ({val_loss:.4f}) aumentou enquanto train_loss ({train_loss:.4f}) diminuiu.")
            if train_loss > 1.0 and val_loss > 1.0:
                print(f"\n⚠️ Underfitting detectado na época {epoch + 1}: Ambos train_loss ({train_loss:.4f}) e val_loss ({val_loss:.4f}) permanecem altos.")
        self.previous_train_loss = train_loss
        self.previous_val_loss = val_loss

    def on_train_begin(self, logs=None):
        self.previous_train_loss = float('inf')
        self.previous_val_loss = float('inf')

# Lista de modelos disponíveis
MODELOS_DISPONIVEIS = [
    ("ResNet50", ResNet50),
    ("ResNet101", ResNet101),
    ("ResNet152", ResNet152),
    ("DenseNet121", DenseNet121),
    ("DenseNet169", DenseNet169),
    ("DenseNet201", DenseNet201),
    ("InceptionV3", InceptionV3),
    ("InceptionResNetV2", InceptionResNetV2),
    ("VGG16", VGG16),
    ("VGG19", VGG19),
    ("MobileNet", MobileNet),
    ("MobileNetV2", MobileNetV2),
    ("MobileNetV3Small", MobileNetV3Small),
    ("MobileNetV3Large", MobileNetV3Large),
    ("NASNetMobile", NASNetMobile),
    ("NASNetLarge", NASNetLarge),
    ("Xception", Xception),
]

# Função para selecionar o modelo
def selecionar_modelo():
    print("\nModelos disponíveis:")
    for i, (nome, _) in enumerate(MODELOS_DISPONIVEIS):
        print(f"{i + 1}. {nome}")
    print(f"{len(MODELOS_DISPONIVEIS) + 1}. Executar todos os modelos")
    escolha = int(input("Selecione o número do modelo desejado: ")) - 1
    if escolha == len(MODELOS_DISPONIVEIS):
        return "all"
    elif 0 <= escolha < len(MODELOS_DISPONIVEIS):
        return [MODELOS_DISPONIVEIS[escolha]]
    else:
        raise ValueError("Opção inválida! Por favor, escolha um número válido.")

# Configurações iniciais
num_classes = 4
batch_size = 16
epochs = 20

# Carregamento dos dados
print("Carregando todas as imagens...")
data_gen = ImageDataGenerator(rescale=1./255)
data_generator = data_gen.flow_from_directory(
    'C:/IA Treinamento/Imagens para treinar',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
print(f"Total de imagens carregadas: {data_generator.samples}")

filenames = data_generator.filenames
labels = data_generator.classes

# Criar grupos (amostras) para evitar data leakage
samples = []
for fname in filenames:
    partes = fname.split('/')
    if partes[0] in ['Limestone', 'Sandstones', 'Shale']:
        # usa a subpasta (amostra) como identificador
        sample_id = partes[1]
    else:
        # "Outros": cada imagem é tratada como amostra independente
        sample_id = partes[-1]
    samples.append(sample_id)

# Split treino+validação vs teste (20% teste)
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_val_idx, test_idx = next(gss.split(filenames, labels, groups=samples))

train_val_files = [filenames[i] for i in train_val_idx]
train_val_labels = [labels[i] for i in train_val_idx]
train_val_groups = [samples[i] for i in train_val_idx]

test_files = [filenames[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]

# Split treino vs validação (20% da parte de treino+validação)
gss_val = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(gss_val.split(train_val_files, train_val_labels, groups=train_val_groups))

train_files = [train_val_files[i] for i in train_idx]
train_labels = [train_val_labels[i] for i in train_idx]

val_files = [train_val_files[i] for i in val_idx]
val_labels = [train_val_labels[i] for i in val_idx]

train_labels = [str(label) for label in train_labels]
val_labels = [str(label) for label in val_labels]
test_labels = [str(label) for label in test_labels]

# Criar dataframes
train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_files, 'class': val_labels})
test_df = pd.DataFrame({'filename': test_files, 'class': test_labels})

# Geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, directory='C:/IA Treinamento/Imagens para treinar',
    x_col='filename', y_col='class', target_size=(224, 224),
    batch_size=batch_size, class_mode='categorical'
)
validation_generator = val_test_datagen.flow_from_dataframe(
    val_df, directory='C:/IA Treinamento/Imagens para treinar',
    x_col='filename', y_col='class', target_size=(224, 224),
    batch_size=batch_size, class_mode='categorical'
)
test_generator = val_test_datagen.flow_from_dataframe(
    test_df, directory='C:/IA Treinamento/Imagens para treinar',
    x_col='filename', y_col='class', target_size=(224, 224),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Seleção do modelo
escolha = selecionar_modelo()
if escolha == "all":
    modelos_para_executar = MODELOS_DISPONIVEIS
else:
    modelos_para_executar = escolha

# Lista para armazenar os resultados
resultados = []

# Iterar sobre os modelos selecionados
for modelo_selecionado, modelo_funcao in modelos_para_executar:
    print(f"\nTreinando com o modelo: {modelo_selecionado}")
    base_model = modelo_funcao(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinamento
    start_time = time.time()
    history = model.fit(
        train_generator, epochs=epochs, validation_data=validation_generator,
        callbacks=[OverUnderfitMonitor()]
    )
    end_time = time.time()

    # Avaliação no conjunto de teste
    print(f"Avaliando o modelo {modelo_selecionado} no conjunto de TESTE...")
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Calcular métricas adicionais
    test_steps = (test_generator.samples + test_generator.batch_size - 1) // test_generator.batch_size
    test_generator.reset()

    y_pred_probs = model.predict(test_generator, steps=test_steps, verbose=1)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = test_generator.classes[:len(y_pred)]

    overall_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    mpca = mean_per_class_accuracy(y_true, y_pred, num_classes=num_classes)

    tempo_total = end_time - start_time

    # Salvar os resultados
    resultados.append({
        "Modelo": modelo_selecionado,
        "Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "MPCA": mpca,
        "Tempo Total (s)": tempo_total
    })

    print(f"\n--- Métricas do Modelo {modelo_selecionado} ---")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    print(f"Precision (weighted): {precision:.2f}")
    print(f"Recall (weighted): {recall:.2f}")
    print(f"Mean Per Class Accuracy (MPCA): {mpca:.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # ===== SALVAR RELATÓRIOS E CURVAS =====
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    base_path = 'C:/IA Treinamento/'

    # 1. Classification report completo
    report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys(), digits=4)
    with open(os.path.join(base_path, f"{modelo_selecionado}_classification_report.txt"), 'w') as f:
        f.write(report)

    # 2. Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=test_generator.class_indices.keys(), columns=test_generator.class_indices.keys())
    cm_df.to_csv(os.path.join(base_path, f"{modelo_selecionado}_confusion_matrix.csv"))

    # 3. Curvas de aprendizado
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(base_path, f"{modelo_selecionado}_learning_curves.csv"), index=False)

    # 4. Gráficos
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{modelo_selecionado} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(base_path, f"{modelo_selecionado}_accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{modelo_selecionado} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(base_path, f"{modelo_selecionado}_loss_curve.png"))
    plt.close()

    # Salvar o modelo
    model.save(f'C:/IA Treinamento/{modelo_selecionado}_model_gpu.h5')
    print(f"\nModelo {modelo_selecionado} salvo com sucesso!\n")
    
    # Liberar memória antes de treinar o próximo modelo
    from tensorflow.keras import backend as K
    import gc
    K.clear_session()
    gc.collect()

# Salvar os resultados no arquivo CSV final
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('C:/IA Treinamento/resultados_modelos.csv', index=False)
print("\nResultados salvos em 'resultados_modelos.csv'")