# 🚀 BlaBli Product Recommendation Chatbot

A full-stack AI-powered product recommendation chatbot for **blabliblulife.com** featuring React frontend and FastAPI backend with RAG-based product search.

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.9+** ([Download Python](https://www.python.org/downloads/))
- **Node.js 18+** and npm ([Download Node.js](https://nodejs.org/))
- **Git** (optional, for cloning the repository)

## 🛠️ Quick Setup

### Option 1: Automated Setup (Windows PowerShell)

1. Navigate to the project directory:
   ```powershell
   cd BLA-BLI-Agent
   ```

2. Run the setup script:
   ```powershell
   .\setup.ps1
   ```

This will automatically:
- Create a Python virtual environment
- Install backend dependencies
- Install frontend dependencies
- Set up configuration files

### Option 2: Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd BLA-BLI-Agent/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd BLA-BLI-Agent/frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

## 🚀 Running the Project

The project consists of two parts that need to run simultaneously: the backend API server and the frontend development server.

### Method 1: Using PowerShell Scripts (Windows)

1. **Start the Backend** (Terminal 1):
   ```powershell
   cd BLA-BLI-Agent
   .\start-backend.ps1
   ```
   The backend will be available at: `http://localhost:8000`
   API documentation: `http://localhost:8000/docs`

2. **Start the Frontend** (Terminal 2):
   ```powershell
   cd BLA-BLI-Agent
   .\start-frontend.ps1
   ```
   The frontend will be available at: `http://localhost:5173`

### Method 2: Manual Start

1. **Start the Backend** (Terminal 1):
   ```bash
   cd BLA-BLI-Agent/backend
   
   # Activate virtual environment
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   # OR
   source venv/bin/activate     # macOS/Linux
   
   # Run the server
   python -m backend.main
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Frontend** (Terminal 2):
   ```bash
   cd BLA-BLI-Agent/frontend
   npm run dev
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5173
   ```

## 📁 Project Structure

```
BBB-Agent/
└── BLA-BLI-Agent/
    ├── backend/              # FastAPI backend
    │   ├── api/              # API endpoints
    │   ├── services/         # Business logic services
    │   ├── models/           # Data models
    │   ├── data/             # Product data and images
    │   ├── main.py           # FastAPI application entry point
    │   └── requirements.txt  # Python dependencies
    │
    ├── frontend/             # React frontend
    │   ├── src/              # Source code
    │   ├── package.json      # Node.js dependencies
    │   └── vite.config.ts    # Vite configuration
    │
    ├── setup.ps1             # Automated setup script
    ├── start-backend.ps1     # Backend start script
    ├── start-frontend.ps1    # Frontend start script
    └── README.md             # Detailed project documentation
```

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in `BLA-BLI-Agent/backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for enhanced LLM responses
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:5173
```

### Frontend Environment Variables

Create a `.env` file in `BLA-BLI-Agent/frontend/` directory:

```env
VITE_API_URL=http://localhost:8000/api
```

## 🧪 Testing the Setup

### Test Backend API

1. Health check:
   ```bash
   curl http://localhost:8000/health
   ```

2. Test chat endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Show me eco-friendly products"}'
   ```

### Test Frontend

Simply open `http://localhost:5173` in your browser and start chatting!

## 📝 Additional Notes

- **Backend runs on:** `http://localhost:8000`
- **Frontend runs on:** `http://localhost:5173`
- **API Documentation:** `http://localhost:8000/docs` (Swagger UI)
- Both servers support hot-reload during development
- Make sure both servers are running before using the application

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'backend'`
- **Solution:** Make sure you're running from the `BLA-BLI-Agent` directory and the virtual environment is activated

**Problem:** Port 8000 already in use
- **Solution:** Change the port in `.env` file or stop the process using port 8000

**Problem:** No products returned
- **Solution:** Ensure `backend/data/products.json` exists. You can run the scraper:
  ```bash
  cd BLA-BLI-Agent/backend
  python -m backend.utils.scraper
  ```

### Frontend Issues

**Problem:** API connection errors
- **Solution:** Ensure backend is running on port 8000 and check `VITE_API_URL` in frontend `.env`

**Problem:** CORS errors
- **Solution:** Verify `FRONTEND_URL` in backend `.env` matches your frontend URL

**Problem:** `npm install` fails
- **Solution:** Try clearing npm cache: `npm cache clean --force` and reinstall

## 📚 More Information

For detailed documentation, architecture details, and development guidelines, see:
- `BLA-BLI-Agent/README.md` - Comprehensive project documentation
- `BLA-BLI-Agent/ARCHITECTURE.md` - System architecture details

## 🎯 Quick Start Summary

```bash
# 1. Setup (one-time)
cd BLA-BLI-Agent
.\setup.ps1

# 2. Start Backend (Terminal 1)
.\start-backend.ps1

# 3. Start Frontend (Terminal 2)
.\start-frontend.ps1

# 4. Open browser
# Navigate to http://localhost:5173
```

---

**Happy Coding! 🎉**

