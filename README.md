# Chanci AI - Employability Score Platform

FastAPI webhook receiver deployed on Vercel Serverless that processes CV submissions from Fillout.com and generates comprehensive employability assessments with job recommendations.

##  Project Overview

**Chanci AI** is an automated employability scoring system that:
-  Receives candidate CV submissions via Fillout.com webhooks
-  Processes CVs using OpenAI GPT-5-nano for intelligent analysis
-  Matches candidates with real jobs using Adzuna API (8 current + 2 future roles)
-  Translates skills into professional corporate terminology using AI
-  Sends results to Webflow CMS for beautiful results pages
-  Generates employability scores (0-100) across 4 dimensions

##  Architecture

```
Fillout.com → Webhook → Vercel (FastAPI) → OpenAI GPT-5-nano → Adzuna API → Webflow CMS
                ↓
         PDF Processing
                ↓
      Skills Translation (GPT-5-nano)
                ↓
    Employability Scoring Algorithm
                ↓
         Results Page Creation
```

### Technology Stack
- **Input**: Fillout.com (forms + PDF storage)
- **Backend**: Python FastAPI on Vercel Serverless Functions
- **AI Processing**: OpenAI GPT-4o (CV analysis), GPT-5-nano (skills translation)
- **Job Matching**: Adzuna API (UK job market)
- **Frontend**: Webflow CMS Collections
- **Infrastructure**: Vercel Serverless, GitHub Actions

##  Project Structure

```
Employability_score/
├── api/
│   ├── __init__.py          # Package marker
│   └── index.py             # Main FastAPI application (1,500+ lines)
├── requirements.txt         # Python dependencies
├── vercel.json              # Vercel deployment configuration
├── .env.example             # Environment variable template
├── .env                     # Local environment variables (git-ignored)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

##  Features

### Employability Scoring Algorithm
The system calculates a comprehensive score (0-100) across 4 dimensions:

1. **CV Quality (30 points)**
   - Professional formatting and structure
   - Clear work history and achievements
   - Proper grammar and language

2. **Skills Match (25 points)**
   - Technical skills relevance
   - Industry-specific expertise
   - Breadth and depth of capabilities

3. **Experience (25 points)**
   - Years of relevant experience
   - Career progression trajectory
   - Project complexity and impact

4. **Personality Fit (20 points)**
   - Working with people (interpersonal skills)
   - Structured task preference (organization)
   - Initiative-taking ability (proactivity)

### AI-Powered CV Analysis
OpenAI GPT-4o extracts:
- Personal details (name, email, phone, LinkedIn)
- Education history (degree, institution, dates, achievements)
- Work experience (roles, companies, dates, responsibilities)
- Technical skills and soft skills
- Projects and certifications
- Career summary and standout achievements

### Job Recommendations
Adzuna API provides 10 real UK jobs:
- **8 Current Jobs**: Match candidate's existing skills (entry-level to mid-level)
- **2 Future Jobs**: Aspirational roles showing career progression
- Each job includes: title, company, location, salary, description, URL, posting date

### Skills Corporate Translation
AI transforms casual/student language into professional terminology:
- "Organised charity events" → "Event Coordination & Cross-functional Collaboration"
- "Good with Excel" → "Data Analysis & Financial Modeling"
- "Team leader in projects" → "Project Management & Team Leadership"

### Webflow CMS Integration
Each submission creates a CMS item with:
- Unique slug: `{submission_id}-{YYYYMMDD}`
- Results URL: `/form-results/form-result-page/{slug}`
- Complete analysis data synced to 10 CMS fields
- Automatic page generation for candidates to view results

## 🛠️ Local Development

### Prerequisites
- Python 3.12+
- Git
- OpenAI API key
- Adzuna API credentials (optional for local testing)
- Webflow API token (optional for local testing)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Employability_score.git
   cd Employability_score
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env and add your API keys:
   # OPENAI_API_KEY=sk-your-actual-key-here
   # ADZUNA_APP_ID=your-adzuna-app-id
   # ADZUNA_APP_KEY=your-adzuna-app-key
   # WEBFLOW_API_TOKEN=your-webflow-token
   # WEBFLOW_COLLECTION_ID=your-collection-id
   ```

5. **Run the development server**
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```

6. **Access the application**
   - API: http://localhost:8000
   - Swagger Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Deployment (Vercel)

### Prerequisites
- GitHub account
- Vercel account
- Repository pushed to GitHub

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy Phase 5: Webflow CMS Integration"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Vercel auto-detects the configuration

3. **Configure Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add all required variables:
     - `OPENAI_API_KEY` (required)
     - `ADZUNA_APP_ID` (required)
     - `ADZUNA_APP_KEY` (required)
     - `WEBFLOW_API_TOKEN` (optional but recommended)
     - `WEBFLOW_COLLECTION_ID` (optional but recommended)
   - Set for "Production" environment
   - Click "Redeploy" to apply changes

4. **Get Your Deployment URL**
   - Vercel provides: `https://your-project.vercel.app`
   - Note this URL for Fillout webhook configuration

### Auto-Deployment
- Every push to `main` branch triggers automatic deployment
- View build logs in Vercel Dashboard
- Monitor function logs in real-time

## API Endpoints

### `GET /`
Root endpoint with service information

**Response**:
```json
{
  "status": "healthy",
  "service": "Chanci AI Webhook Receiver",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "webhook": "/webhook/fillout",
    "debug-tmp": "/debug-tmp",
    "docs": "/docs"
  },
  "environment": {
    "OPENAI_API_KEY": true,
    "ADZUNA_APP_ID": true,
    "ADZUNA_APP_KEY": true,
    "WEBFLOW_API_TOKEN": true,
    "WEBFLOW_COLLECTION_ID": true
  }
}
```


### `POST /webhook/fillout`
Main webhook endpoint to receive Fillout.com submissions

**What it does**:
1. Receives webhook POST request from Fillout
2. Downloads and extracts text from CV PDF
3. Analyzes CV with OpenAI GPT-4o
4. Calculates employability score (0-100)
5. Fetches 10 job recommendations from Adzuna
6. Translates top 3 skills to corporate terminology
7. Sends complete results to Webflow CMS
8. Returns comprehensive JSON response

**Response Structure**:
```json
{
  "status": "success",
  "submission_id": "abc123",
  "candidate": {
    "name": "John Doe",
    "email": "john@example.com"
  },
  "employability_score": {
    "total": 78,
    "cv_quality": 25,
    "skills_match": 20,
    "experience": 18,
    "personality_fit": 15,
    "breakdown": {
      "cv_quality_details": "Well-structured CV...",
      "skills_assessment": "Strong technical skills...",
      "experience_notes": "3 years relevant experience...",
      "personality_insights": "Strong teamwork skills..."
    }
  },
  "cv_analysis": {
    "personal_details": {...},
    "education": [...],
    "work_experience": {...},
    "skills": {
      "technical": ["Python", "SQL", "Excel"],
      "soft": ["Leadership", "Communication"]
    },
    "projects": [...],
    "summary": "Experienced professional..."
  },
  "recommendations": {
    "next_steps": [
      "Your profile is strong! Focus on networking...",
      "Consider specialized certifications...",
      "Practice interview skills..."
    ],
    "suggested_roles": [
      {
        "job_type": "current",
        "title": "Data Analyst",
        "company": "Tech Company Ltd",
        "location": "London",
        "salary": "£30,000 - £40,000",
        "description": "We are seeking...",
        "url": "https://www.adzuna.co.uk/...",
        "posted_date": "2024-12-07T10:00:00Z"
      },
      // ... 7 more current jobs + 2 future jobs
    ]
  },
  "top_skills_corporate": [
    {
      "original": "Team leader in university projects",
      "corporate": "Project Management & Team Leadership",
      "category": "leadership"
    }
  ],
  "webflow_results_url": "/form-results/form-result-page/abc123-20241207",
  "webflow_item_id": "webflow-cms-item-id",
  "processing_time_ms": 5420,
  "errors": []
}
```

## 📧 Contact

For questions or support, contact: Ideatharit@gmail.com

---
