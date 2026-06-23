import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

def create_manuscript_line(text, font_path, out_path):
    font = ImageFont.truetype(font_path, 80)
    
    # Calculate text size
    dummy_img = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Add padding
    pad_x, pad_y = 80, 80
    img_w, img_h = text_w + pad_x*2, text_h + pad_y*2
    
    # Create background (light gray to mimic paper)
    img = Image.new('L', (img_w, img_h), color=240)
    draw = ImageDraw.Draw(img)
    
    # Draw text (dark gray/black)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=40)
    
    # Add a slight blur to simulate ink bleed/camera focus
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    
    img.save(out_path)
    print(f"Saved {out_path}")

def main():
    # Two lines that share the phrases "brown fox" and "over the lazy"
    text_A = "The quick brown fox jumps over the lazy dog"
    text_B = "A wild brown fox runs over the lazy cat"
    
    font_A = r"fonts\Caveat.ttf"
    font_B = r"fonts\DancingScript.ttf"
    
    if not os.path.exists(font_A) or not os.path.exists(font_B):
        print("Fonts not found! Make sure you run this from the project root.")
        return
        
    create_manuscript_line(text_A, font_A, "synthetic_line_A.jpg")
    create_manuscript_line(text_B, font_B, "synthetic_line_B.jpg")

if __name__ == "__main__":
    main()
