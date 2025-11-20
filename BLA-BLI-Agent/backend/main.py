from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
from api.chat import router as chat_router
from services.rag import get_rag_service
from tools import INIT_PUBLIC_DATA

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("[*] Starting BlaBli Product Recommendation API...")
    
    # Initialize RAG service (for backward compatibility)
    rag_service = get_rag_service()
    rag_service.initialize("data/products.json")
    
    print("[+] API ready!")
    
    yield
    
    # Shutdown
    print("[-] Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="BlaBli Product Recommendation API",
    description="AI-powered product recommendation chatbot for blabliblulife.com",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "BlaBli Product Recommendation API",
        "version": "1.0.0"
    }


@app.get("/public-data")
def public_data():
    """For frontend to show products even without chat"""
    return INIT_PUBLIC_DATA


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
