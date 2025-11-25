Orthos ✏️➡️🧊

Turn a quick sketch into a 3D-printable model — in seconds.
Sketching is easy. CAD doesn’t have to be.


⸻

💡 Problem

Anyone can draw an idea on paper. But turning that sketch into a 3D model usually requires hours of CAD learning and complex tools.

Orthos makes it simple end-to-end: draw OR upload → get a 3D model → PRINT.

⸻

🚀 What Orthos Does

Orthos converts a hand-drawn sketch into a ready-to-print STL file using an automated pipeline:
	
	1.	Upload a photo of your drawing.
	
	2.	Clean & interpret the sketch using multimodal AI.
	
	3.	Extract edges and contours with computer vision.
	
	4.	Generate a watertight 3D mesh through extrusion.
	
	5.	Download an STL file for any slicer or 3D printer.

⸻

⚙️ How It Works

1. Frontend (React)

Handles image upload, resizing, and simple cropping.

2. Vision Processing (Python + OpenCV)
	•	Convert to grayscale
	•	Remove noise with blurring
	•	Apply thresholding to isolate the ink lines

How the Image reconstruction algorithm works

2D Pre-processing (The "Mask")

    •    Engine: HTML5 Canvas API (getContext('2d'))
	
 Process:
 
    1.    User upload original image onto an off-screen canvas.
	
    2.    Extracts raw pixel data (getImageData).
	
    3.   Iterates through every pixel, calculates the average grayscale value, and compares it to the user-defined threshold.
	
    4.    Binarization: Marks pixels as strictly "Solid" or "Empty" (Visualized as White vs. Dark Gray).

3. 3D Reconstruction (Voxelization)

    •    Logic: A custom "Pixel-to-Mesh" algorithm inside generateSTL.
    1    Iterates through the 2D binary pixel grid $(x, y)$.
   
    2    Top/Bottom Generation: If a pixel is "Solid," it creates two triangles (a quad) for the floor ($z=0$) and the ceiling ($z=extrusionHeight$)
   
    3    Wall Generation (Neighbor Check): It checks the 4 immediate neighbors (North, South, East, West). If a neighbor is "Empty" (or out of bounds), it generates a vertical wall for that side
   
    4    Output: A massive array of coordinates representing triangular facets.

5. Serialization & Export

    •    Format: ASCII STL (Stereolithography).
   
    •    Process:
   
    1    Converts the array of facets into a specifically formatted string (facet normal... vertex... endfacet).
   
    2    Creates a browser Blob object from the string.
   
    3    Generates a temporary URL (URL.createObjectURL) to trigger a file download.



4. AI Repair & Cleanup

A multimodal model fills in missing edges, closes gaps, and fixes messy contours so the shape can be turned into a solid.

4. Mesh Generation
	•	Convert the cleaned 2D drawing into coordinate arrays
	•	Extrude along the Z-axis to create thickness
	•	Export a binary STL using numpy-stl

5. Validation

STLs were tested on an Anycubic Vyper to ensure the prints slice cleanly and produce solid, watertight geometry.

⸻

What We Learned
	•	Using generative vision models for repair instead of just image creation.
	•	Understanding STL structure, triangle meshes, and normal vectors.
	•	Managing a React ↔ Python architecture for heavy computation.
	•	Iterating fast — we broke plenty of prints before getting stable results.

⸻

Future Improvements
	•	Multi-view modeling: Use front + side sketches for richer shapes.
	•	Curved surfaces: Move past straight extrusions to smooth, organic geometry.
	•	Printer integration: Send G-code directly via OctoPrint or similar APIs.

⸻

Built with 💙 and ☕ by the Wellesley
