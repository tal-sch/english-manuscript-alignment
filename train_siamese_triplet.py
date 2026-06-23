import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IAM_DATA_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data"
SYNTHETIC_DATA_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\Synthetic_Data\words"

class TripletSiameseDataset(Dataset):
    def __init__(self, iam_dir, synthetic_dir, transform=None, epoch_length=20000):
        self.transform = transform
        self.epoch_length = epoch_length
        raw_classes = {}
        
        print("Parsing IAM words.txt...")
        words_txt_path = os.path.join(iam_dir, "ascii", "words.txt")
        if os.path.exists(words_txt_path):
            with open(words_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[1] == 'ok':
                        img_id = parts[0]
                        word_text = " ".join(parts[8:])
                        id_parts = img_id.split('-')
                        if len(id_parts) >= 2:
                            folder1 = id_parts[0]
                            folder2 = f"{id_parts[0]}-{id_parts[1]}"
                            img_path = os.path.join(iam_dir, "words", folder1, folder2, f"{img_id}.png")
                            if os.path.exists(img_path):
                                if word_text not in raw_classes:
                                    raw_classes[word_text] = []
                                raw_classes[word_text].append(img_path)
                                
        print("Parsing Synthetic Data...")
        if os.path.exists(synthetic_dir):
            for word_dir in os.listdir(synthetic_dir):
                full_dir = os.path.join(synthetic_dir, word_dir)
                if os.path.isdir(full_dir):
                    for img_file in os.listdir(full_dir):
                        if img_file.endswith('.png'):
                            if word_dir not in raw_classes:
                                raw_classes[word_dir] = []
                            raw_classes[word_dir].append(os.path.join(full_dir, img_file))
                            
        self.classes = {word: paths for word, paths in raw_classes.items() if len(paths) >= 2}
        self.class_names = list(self.classes.keys())
        
        total_images = sum(len(paths) for paths in self.classes.values())
        print(f"Success! Found {len(self.class_names)} unique English words.")
        print(f"Total usable cropped images for triplets: {total_images}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        try:
            # Choose a random word for the Anchor and Positive
            anchor_word = random.choice(self.class_names)
            anchor_path, positive_path = random.sample(self.classes[anchor_word], 2)
            
            # Choose a different word for the Negative
            negative_word = random.choice(self.class_names)
            while negative_word == anchor_word:
                negative_word = random.choice(self.class_names)
                
            negative_path = random.choice(self.classes[negative_word])
                
            img_a = Image.open(anchor_path).convert("L") 
            img_p = Image.open(positive_path).convert("L")
            img_n = Image.open(negative_path).convert("L")
            
            if self.transform:
                img_a = self.transform(img_a)
                img_p = self.transform(img_p)
                img_n = self.transform(img_n)
                
            return img_a, img_p, img_n
            
        except Exception as e:
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

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System running on: {device}")
    
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    
    siamese_transforms = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])
    
    dataset = TripletSiameseDataset(
        iam_dir=IAM_DATA_DIR, 
        synthetic_dir=SYNTHETIC_DATA_DIR, 
        transform=siamese_transforms, 
        epoch_length=15000  # slightly smaller epoch length since we run for 50 epochs
    )
    train_dataloader = DataLoader(dataset, shuffle=True, num_workers=4, batch_size=BATCH_SIZE)
    
    net = SiameseNetwork().to(device)
    
    # Triplet Margin Loss ensures the positive is closer to the anchor than the negative by at least `margin`
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Phase 2 Training (Siamese Network with Triplet Loss for 50 Epochs)...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        current_loss = 0.0
        
        for i, (img_a, img_p, img_n) in enumerate(train_dataloader):
            img_a, img_p, img_n = img_a.to(device), img_p.to(device), img_n.to(device)
            
            optimizer.zero_grad()
            emb_a = net(img_a)
            emb_p = net(img_p)
            emb_n = net(img_n)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} \t Batch {i} \t Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        avg_loss = current_loss / len(train_dataloader)
        
        print(f"--- Epoch {epoch+1} Completed in {epoch_time:.1f}s | Average Loss: {avg_loss:.4f} ---")
        
        # Save model
        torch.save(net.state_dict(), "siamese_triplet_best.pt")
        
    print("\nTraining Complete! Model saved as 'siamese_triplet_best.pt'")

if __name__ == '__main__':
    main()
