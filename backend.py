# backend.py
import httpx
import aiofiles
import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# -------------------------------
# Config
# -------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Updated ML service endpoint (your deployed model on Render)
ML_SERVICE_URL = "https://ecoscan1.onrender.com/classify_image"

# ✅ Partner mapping – keeps same details
PARTNER_MAP = {
    "cardboard":    {"name": "Local Paper Recycler", "contact": "paper@example.org"},
    "glass":        {"name": "GlassWorks", "contact": "glass@example.org"},
    "metal":        {"name": "Metal Recyclers", "contact": "metal@example.org"},
    "plastic":      {"name": "Plastic Upcycle", "contact": "plastic@example.org"},
    "paper":        {"name": "PaperCycle", "contact": "paper@example.org"},
    "trash":        {"name": "Municipal Waste", "contact": "waste@example.org"},
    "biodegradable": {"name": "Hasiru Dala", "contact": "hasiru@example.org"},
    "recyclable":   {"name": "TerraCycle",  "contact": "terracycle@example.org"},
    "hazardous":    {"name": "Attero Recycling", "contact": "attero@example.org"}
}

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="EcoScan Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Receive an uploaded image, save it, send it to the ML microservice,
    and return the prediction along with the mapped partner.
    """
    try:
        # 1. Save uploaded image
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # 2. Send to ML microservice
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as img:
                files = {"file": (filename, img, "image/jpeg")}
                response = await client.post(ML_SERVICE_URL, files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="ML service failed")

        # Parse the ML service response
        ml_result = response.json()
        predicted_class = ml_result["prediction"]
        confidence = ml_result["probabilities"][predicted_class]

        # 3. Map to partner
        partner = PARTNER_MAP.get(predicted_class, {"name": "Unknown", "contact": "N/A"})

        return {
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "partner": partner
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
