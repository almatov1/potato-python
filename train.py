train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'  # Предполагаем, что есть папка test

# 3. Импорты
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 4. Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_TOP = 15        # Сначала обучаем только верхушку
EPOCHS_FINE_TUNE = 40 # Потом дообучаем всю сеть
OUTPUT_DIR = 'weed_model_stats'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5. Загрузчики данных
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Важно для оценки
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Важно для оценки
)

# 6. Модель
NUM_CLASSES = train_generator.num_classes # Получаем количество классов из генератора
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # сначала замораживаем

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# 7. Компиляция и обучение только верхушки
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Обучаем только верхние слои...")
history_top = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_TOP
)

# 8. Разморозка EfficientNet
print("Размораживаем всю модель...")
base_model.trainable = True

# Обычно уменьшают скорость обучения после разморозки
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 9. Дообучение всей модели
print("Дообучаем всю модель...")
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE
)

# 10. Сохранение
model.save('models/weed_model_efficientnet2.keras')
print("✅ Модель полностью обучена и сохранена!")

# Функция для построения и сохранения ROC Curve
def plot_roc_curve(y_true, y_pred_probs, class_names, filename):
    plt.figure(figsize=(8, 6))
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Класс {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

# Функция для построения и сохранения Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred_probs, class_names, filename):
    plt.figure(figsize=(8, 6))
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        average_precision = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'Класс {class_names[i]} (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()

# 11. Оценка модели на валидационном наборе
print("\nОценка модели на валидационном наборе:")
loss_val, accuracy_val = model.evaluate(val_generator)
print(f"Точность на валидационном наборе: {accuracy_val:.4f}")
print(f"Функция потерь на валидационном наборе: {loss_val:.4f}")

y_true_val = val_generator.classes
y_pred_val_probs = model.predict(val_generator)
y_pred_val = np.argmax(y_pred_val_probs, axis=1)
class_names_val = list(val_generator.class_indices.keys())

print("\nОтчет классификации на валидационном наборе:")
print(classification_report(y_true_val, y_pred_val, target_names=class_names_val))
with open(os.path.join(OUTPUT_DIR, 'classification_report_val.txt'), 'w') as f:
    f.write(classification_report(y_true_val, y_pred_val, target_names=class_names_val))

cm_val = confusion_matrix(y_true_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_val,
            yticklabels=class_names_val)
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Confusion Matrix на валидационном наборе')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_val.png'))
plt.close()

plot_roc_curve(y_true_val, y_pred_val_probs, class_names_val, os.path.join(OUTPUT_DIR, 'roc_curve_val.png'))
plot_precision_recall_curve(y_true_val, y_pred_val_probs, class_names_val, os.path.join(OUTPUT_DIR, 'precision_recall_curve_val.png'))

# 12. Оценка модели на тестовом наборе (если есть тестовые данные)
if test_generator.n > 0:
    print("\nОценка модели на тестовом наборе:")
    loss_test, accuracy_test = model.evaluate(test_generator)
    print(f"Точность на тестовом наборе: {accuracy_test:.4f}")
    print(f"Функция потерь на тестовом наборе: {loss_test:.4f}")

    y_true_test = test_generator.classes
    y_pred_test_probs = model.predict(test_generator)
    y_pred_test = np.argmax(y_pred_test_probs, axis=1)
    class_names_test = list(test_generator.class_indices.keys())

    print("\nОтчет классификации на тестовом наборе:")
    print(classification_report(y_true_test, y_pred_test, target_names=class_names_test))
    with open(os.path.join(OUTPUT_DIR, 'classification_report_test.txt'), 'w') as f:
        f.write(classification_report(y_true_test, y_pred_test, target_names=class_names_test))

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names_test,
                yticklabels=class_names_test)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Confusion Matrix на тестовом наборе')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png'))
    plt.close()

    plot_roc_curve(y_true_test, y_pred_test_probs, class_names_test, os.path.join(OUTPUT_DIR, 'roc_curve_test.png'))
    plot_precision_recall_curve(y_true_test, y_pred_test_probs, class_names_test, os.path.join(OUTPUT_DIR, 'precision_recall_curve_test.png'))

else:
    print("\nТестовый набор не найден или пуст. Пропустил оценку на тестовых данных.")

# 13. Графики обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_top.history['accuracy'] + history_fine.history['accuracy'], label='Точность на обучающем наборе')
plt.plot(history_top.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Точность на валидационном наборе')
plt.title('Точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_plot.png'))
plt.close()

plt.subplot(1, 2, 2)
plt.plot(history_top.history['loss'] + history_fine.history['loss'], label='Функция потерь на обучающем наборе')
plt.plot(history_top.history['val_loss'] + history_fine.history['val_loss'], label='Функция потерь на валидационном наборе')
plt.title('Функция потерь')
plt.xlabel('Эпохи')
plt.ylabel('Значение функции потерь')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
plt.close()

print(f"\nСтатистика и графики сохранены в: {OUTPUT_DIR}")