import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_csv('fasc2_training_losses_0311a.csv')
aunet_df = pd.read_csv('fasc2_training_losses_aunet.csv')

# Extract data for plotting
epochs = df['epoch']
train_iou = df['IoU']
train_accuracy = df['accuracy']
train_loss = df['loss']
val_iou = df['val_IoU']
val_accuracy = df['val_accuracy']
val_loss = df['val_loss']

aunet_epochs = aunet_df['epoch']
aunet_train_iou = aunet_df['IoU']
aunet_train_accuracy = aunet_df['accuracy']
aunet_train_loss = aunet_df['loss']
aunet_val_iou = aunet_df['val_IoU']
aunet_val_accuracy = aunet_df['val_accuracy']
aunet_val_loss = aunet_df['val_loss']

# Plotting
plt.figure(figsize=(8, 6))


# # =================Training=================
# # IoU
# plt.subplot(3, 1, 1)
# plt.plot(epochs, train_iou, label='U-net Training IoU')
# plt.plot(aunet_epochs, aunet_train_iou, label='Attention U-net Training IoU')
# plt.xlabel('Epoch')
# plt.ylabel('IoU')
# plt.ylim(0.80, 1.00)
# plt.title('IoU Over Epochs')
# plt.legend()

# # Accuracy
# plt.subplot(3, 1, 2)
# plt.plot(epochs, train_accuracy, label='U-net Training Accuracy')
# plt.plot(aunet_epochs, aunet_train_accuracy, label='Attention U-net Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim(0.80, 1.00)
# plt.title('Accuracy Over Epochs')
# plt.legend()

# # Loss
# plt.subplot(3, 1, 3)
# plt.plot(epochs, train_loss, label='U-net Training Loss')
# plt.plot(aunet_epochs, aunet_train_loss, label='Attention U-net Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0.00, 0.40)
# plt.title('Loss Over Epochs')
# plt.legend()

# plt.tight_layout()
# plt.show()

# =================Validation=================
# IoU
plt.subplot(3, 1, 1)
plt.plot(epochs, val_iou, label='U-net Validation IoU')
plt.plot(aunet_epochs, aunet_val_iou, label='Attention U-net Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.ylim(0.80, 1.00)
plt.title('IoU Over Epochs')
plt.legend()

# Accuracy
plt.subplot(3, 1, 2)
plt.plot(epochs, val_accuracy, label='U-net Validation Accuracy')
plt.plot(aunet_epochs, aunet_val_accuracy, label='Attention U-net Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.80, 1.00)
plt.title('Accuracy Over Epochs')
plt.legend(loc='center right')

# Loss
plt.subplot(3, 1, 3)
plt.plot(epochs, val_loss, label='U-net Validation Loss')
plt.plot(aunet_epochs, aunet_val_loss, label='Attention U-net Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.00, 0.40)
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

