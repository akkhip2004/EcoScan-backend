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
    "Recyclable":    {"name": "TerraCycle", "contact": "terracycle@example.org"},
    "Biodegradable": {"name": "Hasiru Dala", "contact": "hasiru@example.org"},
    "Non-Recyclable": {"name": "Municipal Waste", "contact": "waste@example.org"},
    "Hazardous":     {"name": "Attero Recycling", "contact": "attero@example.org"}
}

# Map raw ML classes → categories
CATEGORY_MAP = {
    "cardboard": "Recyclable",
    "paper": "Recyclable",
    "plastic": "Recyclable",
    "glass": "Recyclable",
    "metal": "Recyclable",
    "trash": "Non-Recyclable",
    "shoes": "Non-Recyclable",
    "biological": "Biodegradable",
    "organic": "Biodegradable",
    "food": "Biodegradable",
    "hazardous": "Hazardous"
}

# Advice text for each category
ADVICE_MAP = {
    "Recyclable": "Rinse and dry before recycling. Flatten boxes to save space.",
    "Biodegradable": "Compost if possible, otherwise dispose in organic waste bin.",
    "Non-Recyclable": "Dispose in general waste. Avoid mixing with recyclables.",
    "Hazardous": "Handle with care. Dispose through authorized collection centers."
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

# -------------------------------
# Upload & Predict Endpoint
# -------------------------------
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Save uploaded image, send to ML service, return prediction + category + partner.
    """
    try:
        # Save uploaded image
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Send image to ML microservice
        async with aiofiles.open(file_path, "rb") as img:
            content = await img.read()
            files = {"file": (filename, content, file.content_type)}

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(ML_SERVICE_URL, files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ML service failed: {response.text}")

        # Parse ML service response safely
        ml_result = response.json()

        # Compatible with both {"prediction": "x", "confidence": y}
        predicted_class = ml_result.get("prediction")
        confidence = ml_result.get("confidence") or ml_result.get("probabilities", {}).get(predicted_class, None)

        if not predicted_class:
            raise HTTPException(status_code=500, detail="Missing prediction from ML service")

        # Map ML prediction to category and partner
        category = CATEGORY_MAP.get(predicted_class.lower(), "Unknown")
        partner = PARTNER_MAP.get(category, {"name": "Unknown", "contact": "N/A"})
        advice = ADVICE_MAP.get(category, "No advice available for this category.")

        # ✅ Return full structured result (no 'all_predictions')
        return {
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "category": category,
            "partner": partner,
            "advice": advice
        }

    except Exception as e:
        import traceback
        print("⚠️ ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


