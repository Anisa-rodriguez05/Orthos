import numpy as np
from PIL import Image
import io
import os
import json
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, Form, File, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import trimesh
from scipy import ndimage
from skimage import measure
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# --- LOGGING SETUP ---
# Create a custom logger
logger = logging.getLogger("orthos_backend")
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)

# --- CORE LOGIC CLASS (Fallback) ---
class ImageToSTLConverter:
    def __init__(self, 
                 image_file: bytes, 
                 threshold: int = 128, 
                 extrusion_height: float = 5.0, 
                 resolution: float = 0.5, 
                 invert: bool = False):
        self.image_file = image_file
        self.threshold = threshold
        self.extrusion_height = float(extrusion_height)
        self.resolution = float(resolution)
        self.invert = invert

    def process_image(self) -> np.ndarray:
        image_buffer = io.BytesIO(self.image_file)
        image_buffer.seek(0)  # Ensure we're at the start of the buffer
        with Image.open(image_buffer) as img:
            if self.resolution != 1.0:
                new_w = int(img.width * self.resolution)
                new_h = int(img.height * self.resolution)
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
            
            img_gray = img.convert('L')
            data = np.array(img_gray)
            
            if self.invert:
                mask = data < self.threshold
            else:
                mask = data > self.threshold
            return mask

    def generate_stl(self) -> str:
        mask = self.process_image()
        height, width = mask.shape
        
        max_dim = max(width, height)
        scale = 100.0 / max_dim if max_dim > 0 else 1.0
        
        padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        
        facets = []
        z_h = self.extrusion_height

        def add_quad(v1, v2, v3, v4):
            facets.append(
                f"facet normal 0 0 0\n outer loop\n"
                f"  vertex {v1[0]:.4f} {v1[1]:.4f} {v1[2]:.4f}\n"
                f"  vertex {v2[0]:.4f} {v2[1]:.4f} {v2[2]:.4f}\n"
                f"  vertex {v4[0]:.4f} {v4[1]:.4f} {v4[2]:.4f}\n"
                f" endloop\nendfacet"
            )
            facets.append(
                f"facet normal 0 0 0\n outer loop\n"
                f"  vertex {v2[0]:.4f} {v2[1]:.4f} {v2[2]:.4f}\n"
                f"  vertex {v3[0]:.4f} {v3[1]:.4f} {v3[2]:.4f}\n"
                f"  vertex {v4[0]:.4f} {v4[1]:.4f} {v4[2]:.4f}\n"
                f" endloop\nendfacet"
            )

        y_idxs, x_idxs = np.where(mask)
        
        for y, x in zip(y_idxs, x_idxs):
            x1 = x * scale
            y1 = (height - y) * scale
            x2 = (x + 1) * scale
            y2 = (height - y - 1) * scale
            
            add_quad((x1, y1, z_h), (x2, y1, z_h), (x2, y2, z_h), (x1, y2, z_h))
            add_quad((x1, y2, 0), (x2, y2, 0), (x2, y1, 0), (x1, y1, 0))

            px, py = x + 1, y + 1

            if not padded[py-1, px]: 
                add_quad((x1, y1, 0), (x2, y1, 0), (x2, y1, z_h), (x1, y1, z_h))

            if not padded[py+1, px]:
                add_quad((x1, y2, z_h), (x2, y2, z_h), (x2, y2, 0), (x1, y2, 0))

            if not padded[py, px-1]:
                add_quad((x1, y2, 0), (x1, y1, 0), (x1, y1, z_h), (x1, y2, z_h))

            if not padded[py, px+1]:
                add_quad((x2, y1, 0), (x2, y2, 0), (x2, y2, z_h), (x2, y1, z_h))

        return "solid exported\n" + "\n".join(facets) + "\nendsolid exported"

# --- GEMINI-BASED STL PIPELINE ---
class GeminiSTLPipeline:
    """4-Step pipeline for converting images to STL using Gemini AI"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def generate_stl(self, image_bytes: bytes) -> str:
        """Execute the full 4-step pipeline"""
        logger.info("=== STARTING GEMINI STL PIPELINE ===")
        
        # Step 1: Input & Ingestion
        image = self._step1_input_ingestion(image_bytes)
        logger.info("✓ Step 1: Input & Ingestion complete")
        
        # Step 2: 2D Pre-processing (The Mask)
        mask, analysis = await self._step2_preprocessing(image)
        logger.info("✓ Step 2: 2D Pre-processing complete")
        logger.info(f"  Detected: {analysis.get('object_type', 'unknown')} object")
        
        # Step 3: 3D Reconstruction (Voxelization)
        voxels = self._step3_voxelization(mask, analysis)
        logger.info("✓ Step 3: 3D Reconstruction (Voxelization) complete")
        logger.info(f"  Voxel grid: {voxels.shape}")
        
        # Step 4: Serialization & Export
        stl_string = self._step4_export(voxels)
        logger.info("✓ Step 4: Serialization & Export complete")
        logger.info("=== PIPELINE COMPLETE ===")
        
        return stl_string
    
    def _step1_input_ingestion(self, image_bytes: bytes) -> Image.Image:
        """Step 1: Load and validate the input image"""
        try:
            image_buffer = io.BytesIO(image_bytes)
            image_buffer.seek(0)  # Ensure we're at the start of the buffer
            image = Image.open(image_buffer)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    async def _step2_preprocessing(self, image: Image.Image) -> tuple:
        """Step 2: Use Gemini to analyze image and create semantic mask"""
        prompt = """Analyze this technical drawing/image and extract 3D object information.

Return a JSON object with this structure:
{
  "object_type": "cylinder" | "box" | "sphere" | "complex",
  "dimensions": {
    "width": <number in mm>,
    "height": <number in mm>,
    "depth": <number in mm>
  },
  "features": [
    {"type": "hole", "diameter": <number>, "position": {"x": <number>, "y": <number>}},
    {"type": "extrusion", "height": <number>, "shape": "circular" | "rectangular"}
  ],
  "base_shape": {
    "type": "rectangle" | "circle" | "polygon",
    "parameters": {}
  },
  "extrusion_profile": "uniform" | "tapered" | "stepped"
}

Be precise with dimensions. If unclear, estimate based on typical engineering scales."""
        
        try:
            response = self.model.generate_content([prompt, image])
            text = response.text
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            analysis = json.loads(text.strip())
            
            img_array = np.array(image.convert('L'))
            threshold = np.mean(img_array)
            mask = img_array > threshold
            
            return mask, analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}, using fallback")
            img_array = np.array(image.convert('L'))
            mask = img_array > 128
            analysis = {
                "object_type": "box",
                "dimensions": {"width": 100, "height": 100, "depth": 50},
                "features": [],
                "extrusion_profile": "uniform"
            }
            return mask, analysis
    
    def _step3_voxelization(self, mask: np.ndarray, analysis: dict) -> np.ndarray:
        """Step 3: Convert 2D mask + analysis to 3D voxel grid"""
        from skimage.transform import resize
        
        dims = analysis.get('dimensions', {'width': 100, 'height': 100, 'depth': 50})
        depth = int(dims.get('depth', 50))
        
        target_size = (100, 100)
        mask_resized = resize(mask.astype(float), target_size, anti_aliasing=False) > 0.5
        
        height, width = mask_resized.shape
        voxel_depth = max(10, min(depth, 100))
        
        voxels = np.zeros((height, width, voxel_depth), dtype=bool)
        
        profile_type = analysis.get('extrusion_profile', 'uniform')
        
        if profile_type == 'uniform':
            for z in range(voxel_depth):
                voxels[:, :, z] = mask_resized
        
        elif profile_type == 'tapered':
            for z in range(voxel_depth):
                scale_factor = 1.0 - (z / voxel_depth) * 0.5
                scaled_mask = ndimage.zoom(mask_resized.astype(float), scale_factor, order=0)
                h_new, w_new = scaled_mask.shape
                h_offset = (height - h_new) // 2
                w_offset = (width - w_new) // 2
                if h_offset >= 0 and w_offset >= 0:
                    voxels[h_offset:h_offset+h_new, w_offset:w_offset+w_new, z] = scaled_mask > 0.5
        
        else:
            for z in range(voxel_depth):
                voxels[:, :, z] = mask_resized
        
        for feature in analysis.get('features', []):
            if feature['type'] == 'hole':
                pos = feature.get('position', {'x': 50, 'y': 50})
                diameter = feature.get('diameter', 10)
                radius = diameter / 2
                cx, cy = int(pos['x']), int(pos['y'])
                
                for z in range(voxel_depth):
                    for y in range(max(0, cy - int(radius)), min(height, cy + int(radius))):
                        for x in range(max(0, cx - int(radius)), min(width, cx + int(radius))):
                            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                                voxels[y, x, z] = False
        
        return voxels
    
    def _step4_export(self, voxels: np.ndarray) -> str:
        """Step 4: Convert voxel grid to STL mesh"""
        try:
            padded_voxels = np.pad(voxels, 1, mode='constant', constant_values=0)
            
            vertices, faces, normals, values = measure.marching_cubes(
                padded_voxels.astype(float),
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            
            mesh.apply_translation(-mesh.centroid)
            max_extent = max(mesh.extents)
            if max_extent > 0:
                scale_factor = 100.0 / max_extent
                mesh.apply_scale(scale_factor)
            
            export_data = trimesh.exchange.stl.export_stl_ascii(mesh)
            
            if isinstance(export_data, bytes):
                return export_data.decode('utf-8')
            return export_data
            
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")
            raise

async def generate_stl_from_gemini(image_bytes: bytes, api_key: str) -> str:
    """Wrapper function to maintain compatibility"""
    pipeline = GeminiSTLPipeline(api_key)
    return await pipeline.generate_stl(image_bytes)

# --- API SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok", "message": "STL Generator API is running"}

@app.post("/api/generate-stl")
async def generate_stl_endpoint(
    file: UploadFile = File(...),
    threshold: int = Form(128),
    extrusionHeight: float = Form(5.0),
    resolution: float = Form(0.5),
    invert: str = Form("false")
):
    try:
        image_bytes = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
        logger.info(f"Content type: {file.content_type}")
        logger.info(f"First 20 bytes (hex): {image_bytes[:20].hex()}")
        
        if len(image_bytes) == 0:
             logger.warning("Uploaded file is empty")
             return Response(content="Error: Uploaded file is empty", status_code=400)

        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            logger.info("Using Gemini 4-Step Pipeline for generation...")
            try:
                stl_string = await generate_stl_from_gemini(image_bytes, api_key)
            except Exception as e:
                logger.error(f"Gemini pipeline failed: {e}, falling back to algorithmic generation")
                # Check if the error was due to invalid image
                if "cannot identify image file" in str(e):
                    # If Gemini failed because it's not an image, fallback will likely fail too.
                    # But we try fallback anyway, which re-raises the error.
                    pass
                    
                is_inverted = invert.lower() == 'true'
                try:
                    converter = ImageToSTLConverter(
                        image_file=image_bytes,
                        threshold=threshold,
                        extrusion_height=extrusionHeight,
                        resolution=resolution,
                        invert=is_inverted
                    )
                    stl_string = converter.generate_stl()
                except Exception as fallback_e:
                    logger.error(f"Fallback generation failed: {fallback_e}")
                    if "cannot identify image file" in str(fallback_e):
                        return Response(content="Error: Please upload PNG or JPG images only. HEIC format from iPhones is not supported.", status_code=400)
                    raise fallback_e
        else:
            logger.info("No API Key found, using algorithmic generation...")
            is_inverted = invert.lower() == 'true'
            try:
                converter = ImageToSTLConverter(
                    image_file=image_bytes,
                    threshold=threshold,
                    extrusion_height=extrusionHeight,
                    resolution=resolution,
                    invert=is_inverted
                )
                stl_string = converter.generate_stl()
            except Exception as e:
                if "cannot identify image file" in str(e):
                     return Response(content="Error: Please upload PNG or JPG images only. HEIC format from iPhones is not supported.", status_code=400)
                raise e
        
        filename = file.filename.split('.')[0] + "_generated.stl"
        
        return Response(
            content=stl_string,
            media_type="application/vnd.ms-pki.stl",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return Response(content=f"Error generating STL: {str(e)}", status_code=500)

if __name__ == "__main__":
    logger.info("Starting Gemini-powered STL Generator API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
