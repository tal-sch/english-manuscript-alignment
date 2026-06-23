import os
import urllib.request
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torchvision.transforms import v2

OUTPUT_DIR = "Synthetic_Data/words"
FONTS_DIR = "fonts"
WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
WORDS_FILE = "common_words.txt"
NUM_WORDS = 1000
AUGMENTATIONS_PER_FONT = 5

def download_words():
    if not os.path.exists(WORDS_FILE):
        print("Downloading word corpus...")
        urllib.request.urlretrieve(WORDS_URL, WORDS_FILE)
    
    with open(WORDS_FILE, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()][:NUM_WORDS]
    return words

def get_augmentations():
    return v2.Compose([
        v2.RandomRotation(degrees=(-15, 15)),
        v2.ElasticTransform(alpha=25.0, sigma=5.0),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
    ])

def generate_image(word, font_path, font_size=60):
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text bounding box
    dummy_img = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Add padding to allow for rotation and distortion
    pad = 30
    img_w, img_h = text_w + pad*2, text_h + pad*2
    
    # Draw white text on black background to match typical thresholding
    img = Image.new('L', (img_w, img_h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), word, font=font, fill=0)
    
    # Add slight random noise/erosion by applying min filter sometimes
    if random.random() > 0.5:
        img = img.filter(ImageFilter.MinFilter(3))
        
    return img

def main():
    words = download_words()
    font_files = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith('.ttf')]
    
    if not font_files:
        print("No fonts found in 'fonts' directory!")
        return
        
    augmenter = get_augmentations()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_images = len(words) * len(font_files) * AUGMENTATIONS_PER_FONT
    print(f"Generating {total_images} synthetic images...")
    
    count = 0
    for word in words:
        word_dir = os.path.join(OUTPUT_DIR, word)
        os.makedirs(word_dir, exist_ok=True)
        
        for font_file in font_files:
            font_name = os.path.splitext(os.path.basename(font_file))[0]
            
            try:
                base_img = generate_image(word, font_file)
                base_tensor = v2.functional.to_image(base_img)
                base_tensor = v2.functional.to_dtype(base_tensor, torch.float32, scale=True)
                
                for i in range(AUGMENTATIONS_PER_FONT):
                    aug_tensor = augmenter(base_tensor)
                    aug_img = v2.functional.to_pil_image(aug_tensor)
                    
                    # Convert to grayscale mode if not already
                    if aug_img.mode != 'L':
                        aug_img = aug_img.convert('L')
                        
                    out_path = os.path.join(word_dir, f"{font_name}_{i}.png")
                    aug_img.save(out_path)
                    count += 1
            except Exception as e:
                print(f"Error generating '{word}' with {font_name}: {e}")
                
        if count % 1000 == 0:
            print(f"Generated {count}/{total_images} images...")

    print(f"Done! Generated {count} synthetic handwriting images.")

if __name__ == "__main__":
    main()
