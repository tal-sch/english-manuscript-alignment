import os
import urllib.request

FONT_URLS = {
    "Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "DancingScript": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "Pacifico": "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf",
    "Satisfy": "https://github.com/google/fonts/raw/main/apache/satisfy/Satisfy-Regular.ttf",
    "GreatVibes": "https://github.com/google/fonts/raw/main/ofl/greatvibes/GreatVibes-Regular.ttf"
}

def download_fonts(output_dir="fonts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for font_name, url in FONT_URLS.items():
        output_path = os.path.join(output_dir, f"{font_name}.ttf")
        if not os.path.exists(output_path):
            print(f"Downloading {font_name}...")
            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"Saved to {output_path}")
            except Exception as e:
                print(f"Failed to download {font_name}: {e}")
        else:
            print(f"{font_name} already exists.")

if __name__ == "__main__":
    download_fonts()
