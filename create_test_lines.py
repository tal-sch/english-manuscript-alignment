import cv2
from lxml import etree
import os

XML_PATH = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data\xml\a01-000u.xml"
IMG_PATH = r"C:\Users\Tal Sch\Desktop\english_manuscript_alignment\IAM_Data\forms\a01-000u.png"

def get_line_crops():
    if not os.path.exists(IMG_PATH):
        print(f"Error: {IMG_PATH} not found.")
        return
        
    img = cv2.imread(IMG_PATH)
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    
    # Get all line elements
    lines = root.xpath('.//line')
    
    for i, line in enumerate(lines[:2]):  # take first two lines
        # find all words in this line to get the bounding box
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for cmp in line.xpath('.//cmp'):
            x = int(cmp.get('x'))
            y = int(cmp.get('y'))
            w = int(cmp.get('width'))
            h = int(cmp.get('height'))
            x_mins.append(x)
            y_mins.append(y)
            x_maxs.append(x + w)
            y_maxs.append(y + h)
            
        if x_mins:
            # Crop with some padding
            pad = 20
            y1 = max(0, min(y_mins) - pad)
            y2 = min(img.shape[0], max(y_maxs) + pad)
            x1 = max(0, min(x_mins) - pad)
            x2 = min(img.shape[1], max(x_maxs) + pad)
            
            crop = img[y1:y2, x1:x2]
            name = "test_line_A.jpg" if i == 0 else "test_line_B.jpg"
            cv2.imwrite(name, crop)
            print(f"Saved {name}")

if __name__ == '__main__':
    get_line_crops()
