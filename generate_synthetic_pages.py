import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

YOLO_IMAGES_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\YOLO_Dataset\images\train"
YOLO_LABELS_DIR = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\YOLO_Dataset\labels\train"
FONTS_DIR = "fonts"
WORDS_FILE = "common_words.txt"

NUM_PAGES = 500
PAGE_WIDTH = 1200
PAGE_HEIGHT = 1600

def generate_yolo_pages():
    with open(WORDS_FILE, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
        
    font_files = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith('.ttf')]
    
    os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    
    print(f"Generating {NUM_PAGES} synthetic pages...")
    for page_idx in range(NUM_PAGES):
        # Create a new "page" with a light gray/textured background
        bg_color = random.randint(230, 255)
        img = Image.new('L', (PAGE_WIDTH, PAGE_HEIGHT), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        font_path = random.choice(font_files)
        font_size = random.randint(50, 100)
        font = ImageFont.truetype(font_path, font_size)
        
        labels = []
        
        current_y = random.randint(30, 100)
        line_spacing = random.randint(20, 60)
        
        while current_y < PAGE_HEIGHT - 100:
            current_x = random.randint(30, 150)
            word_spacing = random.randint(20, 50)
            
            # Generate a line of text
            while current_x < PAGE_WIDTH - 200:
                word = random.choice(words)
                
                # Get bounding box for the word
                bbox = draw.textbbox((current_x, current_y), word, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                # If word goes off page, stop this line
                if current_x + w > PAGE_WIDTH - 50:
                    break
                    
                # Draw the word
                text_color = random.randint(0, 50)
                draw.text((current_x, current_y), word, font=font, fill=text_color)
                
                # Calculate YOLO coordinates (normalized)
                center_x = (bbox[0] + w/2) / PAGE_WIDTH
                center_y = (bbox[1] + h/2) / PAGE_HEIGHT
                norm_w = w / PAGE_WIDTH
                norm_h = h / PAGE_HEIGHT
                
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                current_x += w + word_spacing
                
            current_y += font_size + line_spacing
            
        # Apply some page-level blur to simulate camera focus
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
            
        img_filename = f"synthetic_page_{page_idx}.jpg"
        label_filename = f"synthetic_page_{page_idx}.txt"
        
        img.save(os.path.join(YOLO_IMAGES_DIR, img_filename))
        with open(os.path.join(YOLO_LABELS_DIR, label_filename), 'w') as f:
            f.write("\n".join(labels))
            
        if (page_idx + 1) % 50 == 0:
            print(f"Generated {page_idx + 1}/{NUM_PAGES} pages...")
            
    print("Virtual Scribe complete! 500 synthetic YOLO pages added to the dataset.")

if __name__ == "__main__":
    generate_yolo_pages()
