import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, GPT2LMHeadModel, GPT2Tokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        caption = self.dataframe.iloc[idx, 1]
        caption_vector = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=25).input_ids.squeeze()
        if self.transform:
            image = self.transform(image)
        return image, caption_vector

    def __len__(self):
        return len(self.dataframe)

def load_and_split_data(image_dir, caption_file, tokenizer):
    data = pd.read_csv(caption_file)
    data.dropna(subset=['caption'], inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = ImageCaptionDataset(train_data, image_dir, tokenizer, transform)
    test_dataset = ImageCaptionDataset(test_data, image_dir, tokenizer, transform)
    return train_dataset, test_dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CaptioningModel(nn.Module):
    def __init__(self, tokenizer):
        super(CaptioningModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = tokenizer

    def forward(self, images, captions):
        features = self.vit(images).last_hidden_state[:,0,:]  # Take the [CLS] token features
        features_repeated = features.unsqueeze(1).repeat(1, captions.size(1), 1)  # Repeat features to match captions length
        outputs = self.gpt2(inputs_embeds=features_repeated, labels=captions).logits  # Use GPT-2 for generating logits
        return outputs


def train_model(model, data_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, captions) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(images, captions)
            outputs = outputs.view(-1, outputs.size(-1))
            captions = captions.view(-1)
            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(data_loader)}, Batch Loss: {loss.item():.4f}")

        average_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1} Completed: Average Loss = {average_loss:.4f}\n')


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    image_dir = '/Users/baristuzemen/Desktop/AIE5601/Project/flick30k/Images'
    caption_file = '/Users/baristuzemen/Desktop/AIE5601/Project/flick30k/captions.txt'
    train_dataset, test_dataset = load_and_split_data(image_dir, caption_file, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    model = CaptioningModel(tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion, epochs=5)  
