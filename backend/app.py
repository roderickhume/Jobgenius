from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from markitdown import MarkItDown
import io
import os
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="JobGenius CL - Backend")

# CORS para permitir llamadas desde el frontend en Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambia a tu dominio frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    prompt_type: str  # "profile", "jobs", "adapt"

@app.post("/api/convert")
async def convert_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(400, detail="Solo se permiten PDF y DOCX")

    try:
        content = await file.read()
        file_stream = io.BytesIO(content)
        
        md = MarkItDown()  # Puedes agregar docintel_endpoint si tienes Azure
        result = md.convert_stream(file_stream, file.filename)
        
        return {
            "success": True,
            "markdown": result.text_content,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar el documento: {str(e)}")


@app.post("/api/ai")
async def call_gemini(request: Request):
    """Proxy seguro para Gemini - oculta la API Key"""
    try:
        body = await request.json()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(500, detail="API Key no configurada")

        # Reenviamos la petición a Gemini
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                json=body,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content=response.json())
            
            return response.json()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
