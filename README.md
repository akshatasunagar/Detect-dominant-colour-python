PROBLEM STATEMENT
The objective of this project is to analyse a given image and extract meaningful colour-based insights.
This includes identifying the dominant colour, generating a colour palette using clustering, calculating HSV statistics, visualising the RGB distribution, and presenting all results in a combined graphical dashboard.
The goal is to help designers, photographers, and developers understand the colour composition of an image efficiently.

1.	APPROACH/METHODOLOGY DATA STRUCTURE USED

	Read the input image and preprocess it by resizing for faster computation.

	Convert the image between BGR, RGB, and HSV colour spaces for different analysis tasks.

	Determine the dominant colour using a frequency counter.

	Use K-Means clustering to generate a 5-colour palette.

	Compute mean Hue, Saturation, and Value from the HSV image.

	Plot RGB intensity histograms to understand brightness and distribution.
	Display all results (image, palette, histogram, colour swatches, and calculated metrics)  in    a clear visual layout.

Data Structures Used
	NumPy Arrays – for image matrices, reshaping, and mathematical operations.
	Python Dictionary – to compute the closest CSS colour name.
	Counter (from collections) – to find the most frequent RGB colour.
	KMeans (sklearn) – to create a 5-cluster colour palette
# Detect-dominant-colour-python
3.	Challenges Faced

	Handling incorrect or invalid image paths given by the user.
	Managing BGR vs RGB confusion due to OpenCV using BGR format.
	Accurately converting arbitrary RGB values to the closest CSS3 color name.
	Balancing runtime while processing large images → solved by resizing.



4.	APPROACH/METHODOLOGY DATA STRUCTURE USED

	Read the input image and preprocess it by resizing for faster computation.
	Convert the image between BGR, RGB, and HSV colour spaces for different analysis tasks.
	Determine the dominant colour using a frequency counter.
	Use K-Means clustering to generate a 5-colour palette.
	Compute mean Hue, Saturation, and Value from the HSV image.
	Plot RGB intensity histograms to understand brightness and distribution.
	Display all results (image, palette, histogram, colour swatches, and calculated metrics)  in a clear visual layout.



5. SCOPE FOR IMPROVEMENT 

	Add an option to save output plots as image files.
	Integrate GUI using Tkinter or PyQt for easier usage.
	Add support for extracting the top N dominant colours instead of only one.
	Improve the closest colour matching using CIEDE2000 instead of Euclidean RGB distance.
	Allow users to choose different clustering algorithms (Mean-Shift, DBSCAN).
	Add contrast, brightness, and saturation analysis.
