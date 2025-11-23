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

3. AI Repair & Cleanup

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
