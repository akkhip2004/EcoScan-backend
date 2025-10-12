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

# ML service URL
ML_SERVICE_URL = "https://ecoscan1.onrender.com/classify_image"

# Partner mapping
PARTNER_MAP = {
    "cardboard":     {"name": "Local Paper Recycler", "contact": "paper@example.org"},
    "glass":         {"name": "GlassWorks", "contact": "glass@example.org"},
    "metal":         {"name": "Metal Recyclers", "contact": "metal@example.org"},
    "plastic":       {"name": "Plastic Upcycle", "contact": "plastic@example.org"},
    "paper":         {"name": "PaperCycle", "contact": "paper@example.org"},
    "trash":         {"name": "Municipal Waste", "contact": "waste@example.org"},
    "biodegradable": {"name": "Hasiru Dala", "contact": "hasiru@example.org"},
    "recyclable":    {"name": "TerraCycle",  "contact": "terracycle@example.org"},
    "hazardous":     {"name": "Attero Recycling", "contact": "attero@example.org"}
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
    Fully async endpoint: save uploaded image, send to ML service, return prediction + partner.
    """
    try:
        # 1️⃣ Save uploaded image asynchronously
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # 2️⃣ Read file asynchronously and send to ML microservice
        async with aiofiles.open(file_path, "rb") as img:
            content = await img.read()
            files = {"file": (filename, content, file.content_type)}

            async with httpx.AsyncClient() as client:
                response = await client.post(ML_SERVICE_URL, files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ML service failed: {response.text}")

        # 3️⃣ Parse ML service response
        ml_result = response.json()
        predicted_class = ml_result["prediction"]
        confidence = ml_result["probabilities"][predicted_class]

        # 4️⃣ Map to partner
        partner = PARTNER_MAP.get(predicted_class, {"name": "Unknown", "contact": "N/A"})

        # 5️⃣ Return result
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
