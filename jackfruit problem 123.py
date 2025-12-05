import cv2
from collections import Counter
import webcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def dcolor(path):
    """Find the dominant color of an image."""
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image. Check the filename or path.")
        return None, None
    image = cv2.resize(image, (150, 150))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image.reshape(-1, 3)
    counts = Counter(map(tuple, pixels))
    dc = counts.most_common(1)[0][0]

    return tuple(map(int, dc)), image


def closest(rcolor):
    """Find the closest CSS3 named color to an RGB tuple."""
    min_colors = {}
    for cname, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - rcolor[0]) ** 2
        gd = (g_c - rcolor[1]) ** 2
        bd = (b_c - rcolor[2]) ** 2
        min_colors[(rd + gd + bd)] = cname
    return min_colors[min(min_colors.keys())]


def epalette(image, n_colors=5):
    """Extract a color palette using K-Means clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)
    return palette


def histogram(image):
    """Plot RGB intensity histograms."""
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist.ravel(), color=col)


def analyze(image):
    """Calculate mean HSV values."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hmean = np.mean(hsv[:, :, 0]) * 360 / 180
    smean = np.mean(hsv[:, :, 1]) * 100 / 255
    vmean = np.mean(hsv[:, :, 2]) * 100 / 255
    return hmean, smean, vmean


# ---------------------------
#   MAIN PROGRAM
# ---------------------------
if _name_ == "_main_":
    path = input("Enter the image filename or full path: ")
    dc, small = dcolor(path)

    if dc:
        # Hex code
        try:
            chex = webcolors.rgb_to_hex(dc)
        except Exception:
            chex = "#%02x%02x%02x" % dc

        # Closest named color
        try:
            cname = closest(dc)
        except Exception:
            cname = "No close match found"

        # HSV + Palette
        hmean, smean, vmean = analyze(small)
        palette = epalette(small, n_colors=5)

        # Console output
        print("\nDominant Color (RGB):", dc)
        print("Dominant Hex Code:", chex)
        print("Closest Named Color:", cname)
        print(f"HSV Means → H:{hmean:.0f}°, S:{smean:.0f}%, V:{vmean:.0f}%")
        print("Extracted Palette (RGB):", palette)

        # ---------------------------
        #      VISUAL DISPLAY
        # ---------------------------
        plt.figure(figsize=(15, 12), facecolor='#f0f0f0')

        # Input image
        ax1 = plt.axes([0.05, 0.55, 0.45, 0.4])
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.set_title("Input Image", fontsize=14)
        ax1.axis("off")

        # Text summary
        ax = plt.axes([0.55, 0.75, 0.4, 0.2])
        summary = (
            f"Color Analysis Summary\n"
            f"• Dominant RGB: {dc}\n"
            f"• Dominant Hex: {chex}\n"
            f"• Closest Name: {cname}\n"
            f"• HSV Means: H:{hmean:.0f}° | S:{smean:.0f}% | V:{vmean:.0f}%\n"
        )
        ax.text(0.01, 1, summary, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', linespacing=1.5)
        ax.axis('off')

        # RGB pie chart
        ax2 = plt.axes([0.57, 0.56, 0.15, 0.15])
        total = np.sum(dc)
        percentage = [(channel / total) * 100 for channel in dc]
        colorsp = ['#ff0000', '#00ff00', '#0000ff']
        labelsp = [f'Red {percentage[0]:.1f}%',
                   f'Green {percentage[1]:.1f}%',
                   f'Blue {percentage[2]:.1f}%']
        ax2.pie(percentage, labels=labelsp,
                autopct="", colors=colorsp, startangle=90)
        ax2.set_title("RGB Composition", fontsize=10)

        # Dominant color swatch
        ax3 = plt.axes([0.75, 0.56, 0.15, 0.15])
        swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        swatch[:] = dc
        ax3.imshow(swatch)
        ax3.set_title(f"Dominant Color\n{chex}", fontsize=10)
        ax3.axis("off")

        # Palette swatches
        ax4 = plt.axes([0.05, 0.40, 0.9, 0.05])
        pimg = np.zeros((50, 50 * len(palette), 3), dtype=np.uint8)
        for i, col in enumerate(palette):
            pimg[:, i*50:(i+1)*50] = np.array(col, dtype=np.uint8)
        ax4.imshow(pimg, aspect='auto')
        ax4.set_title("Extracted Palette (K-Means)", fontsize=14, pad=10)
        ax4.axis("off")

        # Histogram
        ax5 = plt.axes([0.05, 0.05, 0.9, 0.30])
        histogram(small)
        ax5.set_title("RGB Intensity Histogram", fontsize=14)
        ax5.set_xlabel("Pixel Value (0-255)")
        ax5.set_ylabel("Frequency (Pixels)")
        ax5.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()
