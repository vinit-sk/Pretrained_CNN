from torchvision import models
import torch.nn as nn
from dataloader import load_train_val_dataset
import torch
import torch.optim as optim
import os
from torchsummary import summary
from tqdm import tqdm

data_root = '/root/VINIT/Dataset/'
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(data_root):
    train_loader, val_loader, class_names = load_train_val_dataset(data_root, batch_size)
    num_classes = len(class_names)    
    # Device config
    
    # Load pretrained model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    #to print model summary uncomment and run
    # summary(model,(3,224,224)) #model and model's input shape
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #optimize lr value
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0
    num_epochs = 50
    #for graphs
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, prediction = torch.max(outputs,1)
            train_total += labels.shape[0]
            train_correct = train_correct + (prediction==labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = train_correct/train_total
        # Validation
        model.eval()
        val_total = 0
        val_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, prediction = torch.max(outputs,1)
                val_total += labels.shape[0]
                val_correct = val_correct + (prediction==labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_acc = val_correct/val_total
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train acc: {train_acc*100:.2f}%, Val acc: {val_acc*100:.2f}% Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = {'state_dict' : model.state_dict(),
                          'class_names': class_names}
            torch.save(state_dict, 'best_model_basic_val.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break 
    # Plot and save Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png')
    plt.close()  # Close the figure to avoid overlap with next plot
    
    # Plot and save Loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.close()
else:
    print('Incorrect Path :',data_root)

