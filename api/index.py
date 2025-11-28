"""
Chanci AI - CV Processing Webhook Receiver
FastAPI application deployed on Vercel Serverless Functions
Receives webhooks from Fillout.com with candidate CV submissions
"""

# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import httpx
from pypdf import PdfReader
from io import BytesIO
from openai import OpenAI
import time


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Chanci AI Webhook Receiver",
    description="Receives and processes CV submissions from Fillout.com",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# CORS Middleware - Allow Fillout.com to send webhooks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Fillout.com domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS (Data Validation)
# ============================================================================

class CandidateData(BaseModel):
    """
    Model for candidate information from Fillout form
    Fields are optional since we don't know exact webhook structure yet
    """
    submissionId: Optional[str] = Field(None, description="Unique submission ID")
    formId: Optional[str] = Field(None, description="Fillout form ID")
    status: Optional[str] = Field(None, description="Submission status")

    # Candidate personal info
    fullName: Optional[str] = Field(None, alias="Full Name:")
    email: Optional[str] = Field(None, alias="Email:")
    phoneNumber: Optional[str] = Field(None, alias="Phone Number:")
    dateOfBirth: Optional[str] = Field(None, alias="Date of Birth:")
    linkedIn: Optional[str] = Field(None, alias="Add your LinkedIn")

    # Skills and experience
    basicSkills: Optional[str] = Field(None, alias="Basic Skills")
    otherSkills: Optional[str] = Field(None, alias="Other Skills")
    experienceLevel: Optional[str] = Field(None, alias="Experience Level")
    softSkills: Optional[str] = Field(None, alias="Soft Skills")

    # Personality traits
    enjoysWorkingWithPeople: Optional[str] = None
    prefersClearStructure: Optional[str] = None
    takesInitiative: Optional[str] = None

    # CV file - could be URL, base64, or file object
    cvFile: Optional[Any] = Field(None, alias="Please upload your CV to get started")

    # Metadata
    submissionUrl: Optional[str] = Field(None, alias="Url")
    lastUpdated: Optional[str] = Field(None, alias="Last updated")

    class Config:
        populate_by_name = True  # Allow both alias and field name


class WebhookPayload(BaseModel):
    """
    Flexible model for Fillout webhook payload
    Structure will be discovered from first real webhook
    """
    data: Optional[Dict[str, Any]] = None
    submission: Optional[Dict[str, Any]] = None
    questions: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"  # Allow additional fields not defined


class FilloutWebhookPayload(BaseModel):
    """
    Model for actual Fillout webhook payload structure
    Based on real webhook data received
    """
    SubmissionID: str
    BasicSkills: List[str]
    OtherSkills: str
    ExperienceLvl: str
    SoftSkills: List[str]
    People: int = Field(ge=1, le=5)
    StructuredTask: int = Field(ge=1, le=5)
    FullName: str
    DoB: str
    Email: str
    PhoneNo: str
    Linkedin: str
    CV: List[Dict[str, str]]

    class Config:
        extra = "allow"  # Allow additional fields


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_webhook_data(
    event_type: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Comprehensive logging function for webhook events
    Logs to stdout - Vercel captures and displays in Dashboard

    Args:
        event_type: Type of event (e.g., "webhook_received", "error")
        data: Payload data to log
        headers: Request headers
        extra_info: Additional information to log
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "headers": {
            "content-type": headers.get("content-type"),
            "user-agent": headers.get("user-agent"),
            "x-forwarded-for": headers.get("x-forwarded-for"),
            # Add more headers as needed
        },
        "payload_structure": {
            "top_level_keys": list(data.keys()) if isinstance(data, dict) else "not_a_dict",
            "payload_size_bytes": len(str(data)),
        },
        "payload": data,  # Full payload for analysis
    }

    # Add extra info if provided
    if extra_info:
        log_entry["extra_info"] = extra_info

    # Extract CV/PDF information if present
    cv_info = extract_cv_metadata(data)
    if cv_info:
        log_entry["cv_metadata"] = cv_info

    # Print as formatted JSON (Vercel logs capture this)
    print("=" * 80)
    print(f"WEBHOOK EVENT: {event_type}")
    print("=" * 80)
    print(json.dumps(log_entry, indent=2, default=str))
    print("=" * 80)


def extract_cv_metadata(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract CV/PDF file metadata from webhook payload
    Handles multiple possible structures:
    - URL to PDF file
    - Base64 encoded PDF
    - File object with metadata

    Args:
        data: Webhook payload data

    Returns:
        Dictionary with CV metadata or None if not found
    """
    cv_info = {}

    # Search for CV in various possible locations
    possible_keys = [
        "Please upload your CV to get started",
        "cvFile",
        "cv_file",
        "file",
        "files",
        "attachment",
        "attachments"
    ]

    for key in possible_keys:
        if key in data:
            cv_data = data[key]

            # If it's a string, might be a URL
            if isinstance(cv_data, str):
                cv_info = {
                    "found": True,
                    "location": key,
                    "type": "url" if cv_data.startswith("http") else "string",
                    "value": cv_data[:200] + "..." if len(cv_data) > 200 else cv_data,
                    "length": len(cv_data)
                }
                break

            # If it's a dict, might have file metadata
            elif isinstance(cv_data, dict):
                cv_info = {
                    "found": True,
                    "location": key,
                    "type": "object",
                    "keys": list(cv_data.keys()),
                    "data": cv_data
                }
                break

            # If it's a list, might have multiple files
            elif isinstance(cv_data, list) and len(cv_data) > 0:
                cv_info = {
                    "found": True,
                    "location": key,
                    "type": "array",
                    "count": len(cv_data),
                    "first_item": cv_data[0] if cv_data else None
                }
                break

    return cv_info if cv_info else None


def get_environment_status() -> Dict[str, bool]:
    """
    Check which environment variables are configured
    NEVER log actual values - only check presence

    Returns:
        Dictionary with boolean status of each env var
    """
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ADZUNA_API_KEY": bool(os.environ.get("ADZUNA_API_KEY")),
        "WEBFLOW_API_KEY": bool(os.environ.get("WEBFLOW_API_KEY")),
    }


async def download_pdf(url: str) -> Optional[bytes]:
    """
    Download PDF from Fillout S3 URL

    Args:
        url: S3 URL to PDF file

    Returns:
        PDF file as bytes, or None if download fails
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            print(f"PDF downloaded successfully - Size: {len(response.content)} bytes")
            return response.content

    except httpx.TimeoutException:
        print(f"Timeout downloading PDF from {url}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"HTTP error downloading PDF: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None


def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract text content from PDF bytes

    Args:
        pdf_bytes: PDF file as bytes

    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)

        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        print(f"Text extracted from {len(reader.pages)} pages - Total length: {len(full_text)} chars")

        return full_text if full_text.strip() else None

    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None


async def analyze_cv_with_openai(cv_text: str, candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyze CV content using OpenAI GPT-4o

    Args:
        cv_text: Extracted text from CV
        candidate_data: Form data from webhook (name, skills, experience, etc.)

    Returns:
        Structured analysis with work experience, education, skills, strengths, improvements, and CV quality score
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not configured")
            return None

        client = OpenAI(api_key=api_key)

        system_prompt = "You are an expert career counselor and CV analyst. Analyze the provided CV and candidate information to assess employability."

        user_prompt = f"""Analyze this CV and provide structured feedback:

CV Content:
{cv_text[:4000]}

Candidate Information:
- Name: {candidate_data.get('FullName', 'N/A')}
- Experience Level: {candidate_data.get('ExperienceLvl', 'N/A')}
- Basic Skills: {', '.join(candidate_data.get('BasicSkills', []))}
- Other Skills: {candidate_data.get('OtherSkills', 'N/A')}
- Soft Skills: {', '.join(candidate_data.get('SoftSkills', []))}

Provide:
1. Work experience summary (summary text, years of experience, roles)
2. Education summary (highest level, field, institutions)
3. Technical skills identified from CV
4. Soft skills identified from CV
5. Career level assessment (entry/mid/senior)
6. Key strengths (3-5 points)
7. Areas for improvement (3-5 points)
8. CV quality score (0-100) based on completeness, clarity, and professionalism

Return as JSON with this exact structure:
{{
  "work_experience": {{
    "summary": "brief summary",
    "years": 0,
    "roles": ["role1", "role2"]
  }},
  "education": {{
    "highest_level": "degree level",
    "field": "field of study",
    "institutions": ["institution1"]
  }},
  "skills": {{
    "technical": ["skill1", "skill2"],
    "soft": ["skill1", "skill2"]
  }},
  "career_level": "entry",
  "strengths": ["strength1", "strength2"],
  "improvements": ["improvement1", "improvement2"],
  "cv_quality_score": 85
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1500
        )

        analysis = json.loads(response.choices[0].message.content)
        print(f"OpenAI analysis completed - CV Quality Score: {analysis.get('cv_quality_score', 'N/A')}")

        return analysis

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None


def calculate_employability_score(openai_analysis: Optional[Dict[str, Any]], form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate employability score (0-100) combining OpenAI analysis and form responses

    Scoring breakdown:
    - CV Quality: 30 points (from OpenAI)
    - Skills Match: 25 points (from form + CV)
    - Experience Level: 25 points
    - Personality Fit: 20 points (People + StructuredTask scores)

    Args:
        openai_analysis: Analysis results from OpenAI (or None if failed)
        form_data: Form submission data

    Returns:
        Dictionary with total score, breakdown, grade, and percentile
    """
    breakdown = {
        "cv_quality": 0,
        "skills_match": 0,
        "experience": 0,
        "personality_fit": 0
    }

    # 1. CV Quality Score (0-30 points)
    if openai_analysis and "cv_quality_score" in openai_analysis:
        cv_quality = openai_analysis["cv_quality_score"]
        breakdown["cv_quality"] = int((cv_quality / 100) * 30)
    else:
        breakdown["cv_quality"] = 15

    # 2. Skills Match (0-25 points)
    basic_skills = form_data.get("BasicSkills", [])
    other_skills = form_data.get("OtherSkills", "")
    soft_skills = form_data.get("SoftSkills", [])

    skills_count = len(basic_skills) + len(soft_skills)
    if other_skills and other_skills.strip():
        skills_count += len(other_skills.split(","))

    breakdown["skills_match"] = min(25, skills_count * 3)

    # 3. Experience Level (0-25 points)
    experience_mapping = {
        "Just starting out": 5,
        "Some experience": 12,
        "Experienced": 18,
        "Very experienced": 25
    }
    experience_level = form_data.get("ExperienceLvl", "Just starting out")
    breakdown["experience"] = experience_mapping.get(experience_level, 10)

    # 4. Personality Fit (0-20 points)
    people_score = form_data.get("People", 3)
    structured_score = form_data.get("StructuredTask", 3)
    breakdown["personality_fit"] = int(((people_score + structured_score) / 10) * 20)

    total_score = sum(breakdown.values())

    # Calculate grade
    if total_score >= 90:
        grade = "A+"
    elif total_score >= 80:
        grade = "A"
    elif total_score >= 70:
        grade = "B+"
    elif total_score >= 60:
        grade = "B"
    elif total_score >= 50:
        grade = "C+"
    else:
        grade = "C"

    # Estimate percentile
    percentile = min(99, int((total_score / 100) * 100))

    return {
        "total": total_score,
        "breakdown": breakdown,
        "grade": grade,
        "percentile": percentile
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - Health check and service info
    """
    return {
        "status": "healthy",
        "service": "Chanci AI Webhook Receiver",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook/fillout",
            "docs": "/docs"
        },
        "environment": get_environment_status()
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "webhook-receiver",
        "environment_configured": get_environment_status()
    }


@app.post("/webhook/fillout")
async def receive_fillout_webhook(request: Request):
    """
    Main webhook endpoint to receive Fillout.com submissions

    This endpoint:
    1. Receives the webhook POST request
    2. Extracts headers and body
    3. Logs complete payload structure
    4. Extracts CV/PDF information
    5. Returns 200 OK acknowledgment

    Future enhancements:
    - Parse CV with OpenAI
    - Match jobs with Adzuna
    - Send results to Webflow
    """
    try:
        # Get request headers
        headers = dict(request.headers)

        # Get raw body
        body = await request.body()

        # Try to parse as JSON
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            # If not JSON, log raw body
            log_webhook_data(
                event_type="webhook_received_non_json",
                data={"raw_body": body.decode()[:1000]},  # First 1000 chars
                headers=headers,
                extra_info={"error": "Could not parse as JSON"}
            )
            return JSONResponse(
                status_code=200,
                content={"status": "received", "message": "Non-JSON payload logged"}
            )

        # Log the complete webhook payload
        log_webhook_data(
            event_type="webhook_received",
            data=payload,
            headers=headers,
            extra_info={
                "payload_keys_count": len(payload.keys()) if isinstance(payload, dict) else 0
            }
        )

        # Extract specific fields we care about
        submission_id = None
        candidate_email = None

        # Try to find submission ID (field name might vary)
        for key in ["submissionId", "submission_id", "id", "Submission ID"]:
            if key in payload:
                submission_id = payload[key]
                break

        # Try to find email
        for key in ["email", "Email:", "Email"]:
            if isinstance(payload.get("data"), dict):
                candidate_email = payload["data"].get(key)
            if not candidate_email and key in payload:
                candidate_email = payload[key]
            if candidate_email:
                break

        # Return success response
        response_data = {
            "status": "success",
            "message": "Webhook received and logged",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "submission_id": submission_id,
            "candidate_email": candidate_email,
            "next_steps": "CV processing will be implemented in Phase 2"
        }

        print(f"✅ Webhook processed successfully - Submission ID: {submission_id}")

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        # Log error but still return 200 to webhook provider
        # We don't want Fillout to retry due to our errors
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        log_webhook_data(
            event_type="webhook_error",
            data={"error": error_info},
            headers=dict(request.headers),
            extra_info={"exception": str(e)}
        )

        print(f"Error processing webhook: {str(e)}")

        # Return 200 to prevent retries
        return JSONResponse(
            status_code=200,
            content={
                "status": "error_logged",
                "message": "Error occurred but logged for investigation"
            }
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions gracefully
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler
    """
    print(f"Unhandled exception: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


# ============================================================================
# VERCEL SERVERLESS HANDLER
# ============================================================================

# Export the FastAPI app for Vercel
# Vercel looks for 'app' variable in api/index.py

