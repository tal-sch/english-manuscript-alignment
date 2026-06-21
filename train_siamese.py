import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data"

class IAMSiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None, epoch_length=10000):
        self.data_dir = data_dir
        self.transform = transform
        self.epoch_length = epoch_length
        
        words_txt_path = os.path.join(data_dir, "ascii", "words.txt")
        if not os.path.exists(words_txt_path):
            # Check the old kaggle location if not found in ascii
            words_txt_path = os.path.join(data_dir, "words.txt")
            
        raw_classes = {}
        
        print("Parsing IAM words.txt...")
        with open(words_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                # skip comments
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 9:
                    img_id = parts[0]
                    status = parts[1]
                    word_text = " ".join(parts[8:])
                    
                    if status != 'ok':
                        continue
                        
                    # Calculate path
                    id_parts = img_id.split('-')
                    if len(id_parts) >= 2:
                        folder1 = id_parts[0]
                        folder2 = f"{id_parts[0]}-{id_parts[1]}"
                        img_path = os.path.join(data_dir, "words", folder1, folder2, f"{img_id}.png")
                        
                        if os.path.exists(img_path):
                            if word_text not in raw_classes:
                                raw_classes[word_text] = []
                            raw_classes[word_text].append(img_path)
                            
        # Filter to keep only words with at least 2 examples
        self.classes = {word: paths for word, paths in raw_classes.items() if len(paths) >= 2}
        self.class_names = list(self.classes.keys())
        
        total_images = sum(len(paths) for paths in self.classes.values())
        print(f"Success! Found {len(self.class_names)} unique English words with multiple examples.")
        print(f"Total usable cropped images for pairs: {total_images}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        try:
            # 50% chance for a positive pair
            is_positive = random.random() > 0.5
            
            if is_positive:
                chosen_word = random.choice(self.class_names)
                img1_path, img2_path = random.sample(self.classes[chosen_word], 2)
                label = torch.tensor([1.0], dtype=torch.float32)
            else:
                word1, word2 = random.sample(self.class_names, 2)
                img1_path = random.choice(self.classes[word1])
                img2_path = random.choice(self.classes[word2])
                label = torch.tensor([0.0], dtype=torch.float32)
                
            img1 = Image.open(img1_path).convert("L") 
            img2 = Image.open(img2_path).convert("L")
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                
            return img1, img2, label
            
        except Exception as e:
            # Recursive retry if an image is corrupted or missing
            return self.__getitem__(random.randint(0, self.epoch_length - 1))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System running on: {device}")
    
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    
    siamese_transforms = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])
    
    dataset = IAMSiameseDataset(data_dir=DATA_DIR, transform=siamese_transforms, epoch_length=20000)
    train_dataloader = DataLoader(dataset, shuffle=True, num_workers=4, batch_size=BATCH_SIZE)
    
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Phase 3 Training (Siamese Network)...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        current_loss = 0.0
        
        for i, (img1, img2, label) in enumerate(train_dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output1, output2 = net(img1, img2)
            loss_contrastive = criterion(output1, output2, label)
            
            loss_contrastive.backward()
            optimizer.step()
            
            current_loss += loss_contrastive.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} \t Batch {i} \t Loss: {loss_contrastive.item():.4f}")
                
        epoch_time = time.time() - start_time
        avg_loss = current_loss / len(train_dataloader)
        
        print(f"--- Epoch {epoch+1} Completed in {epoch_time:.1f}s | Average Loss: {avg_loss:.4f} ---")
        
        # Save model after each epoch in case of interruption
        torch.save(net.state_dict(), "siamese_iam_best.pt")
        
    print("\nTraining Complete! Model saved as 'siamese_iam_best.pt'")

if __name__ == '__main__':
    main()
