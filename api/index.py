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
import hmac
import hashlib

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
    fullName: str
    email: str
    phone: str
    linkedin: str
    skills: List[str]
    otherSkills: str
    experience: str
    softSkills: List[str]
    workingWithPeople: int = Field(ge=1, le=5)
    clearStructure: int = Field(ge=1, le=5)
    takingInitiative: int = Field(ge=1, le=5)
    cvFileName: List[Dict[str, str]]
    submittedAt: str

    class Config:
        extra = "allow"  # Allow additional fields


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
    InitiativeTask: int = Field(ge=1, le=5)
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
    
def verify_webflow_signature(body_bytes: bytes, headers: dict, secret: str) -> bool:
    """
    Verifies that the webhook actually came from Webflow.
    
    Webflow Signatures format:
    x-webflow-signature: hash
    x-webflow-timestamp: 123456789
    
    Verification = HMAC_SHA256(secret, timestamp + ":" + raw_body)
    """
    try:
        # 1. Get headers
        timestamp = headers.get("x-webflow-timestamp")
        signature = headers.get("x-webflow-signature")
        
        if not timestamp or not signature:
            print("Missing Webflow verification headers")
            return False

        # 2. Check timestamp freshness (Optional but recommended: prevents replay attacks)
        request_time = int(timestamp) / 1000
        if (time.time() - request_time) > 300: # 5 minutes
            print("Webflow request is too old (potential replay attack)")
            return False

        # 3. Create the string to sign: timestamp + ":" + raw_body_string
        string_to_sign = f"{timestamp}:".encode("utf-8") + body_bytes

        # 4. Calculate expected signature
        expected_signature = hmac.new(
            key=secret.encode("utf-8"),
            msg=string_to_sign,
            digestmod=hashlib.sha256
        ).hexdigest()

        # 5. Compare signatures safely
        return hmac.compare_digest(expected_signature, signature)

    except Exception as e:
        print(f"Signature verification error: {e}")
        return False


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
        "ADZUNA_APP_ID": bool(os.environ.get("ADZUNA_APP_ID")),
        "ADZUNA_APP_KEY": bool(os.environ.get("ADZUNA_APP_KEY")),
        "WEBFLOW_API_TOKEN": bool(os.environ.get("WEBFLOW_API_TOKEN")),
        "WEBFLOW_COLLECTION_ID": bool(os.environ.get("WEBFLOW_COLLECTION_ID")),
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
            response = await client.get(url, follow_redirects=True)
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
5. Career level assessment (graduate/entry/mid/senior)
6. Key strengths (3-5 points)
7. Areas for improvement (3-5 points)
8. CV quality score (0-100) based on completeness, clarity, and professionalism
9. Suggested job roles (List of 3 specific job titles best suited for profile's skills) - short and simple title allowing for Adzuna API job search (No bracket answer)


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
  "cv_quality_score": number between 0-100,
  "suggested_job_roles": ["Role 1", "Role 2", "Role 3] ex. AI Engineer, Python Developer, Graduate Software Engineer
}}"""
        # Planned changes to model gpt-5-nano
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            #temperature=0.7,
            #max_tokens=15000    
        )

        analysis = json.loads(response.choices[0].message.content)
        print(f"OpenAI analysis completed - CV Quality Score: {analysis.get('cv_quality_score', 'N/A')}")

        return analysis

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None
    
# ============================================================================
# ADZUNA JOB SEARCH FUNCTIONS (Phase 3)
# ============================================================================

def extract_current_keywords(candidate_data: Dict[str, Any], cv_analysis: Optional[Dict[str, Any]]) -> List[str]:
    """
    Extract keywords for current job search from candidate data and CV analysis

    Args:
        candidate_data: Form submission data
        cv_analysis: OpenAI CV analysis results

    Returns:
        List of keywords prioritized for current job search
    """
    keywords = []
    
    # Priority 1: Job roles recommendation from LLM
    # These are specific titles like "AI Engineer" derived from the CV content
    if cv_analysis and "suggested_job_roles" in cv_analysis:
        suggested_roles = cv_analysis.get("suggested_job_roles", [])
        keywords.extend(suggested_roles)
    
    # Priority 2: Other skills (comma-separated)
    other_skills = candidate_data.get("OtherSkills", "")
    if other_skills:
        skills_list = [s.strip() for s in other_skills.split(",") if s.strip()]
        keywords.extend(skills_list[:2])

    # Priority 3: Technical skills from CV
    if cv_analysis and "skills" in cv_analysis:
        tech_skills = cv_analysis["skills"].get("technical", [])
        keywords.extend(tech_skills[:3])
        
    # Priority 4: Basic skills from form
    basic_skills = candidate_data.get("BasicSkills", [])
    keywords.extend(basic_skills)

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in seen:
            seen.add(keyword_lower)
            unique_keywords.append(keyword)

    return unique_keywords[:5]  # Limit to top 5


# Swap the roles to be a +1 position
def extract_future_keywords(candidate_data: Dict[str, Any], current_keywords: List[str]) -> List[str]:
    """
    Generate keywords for future/advanced job search

    Args:
        candidate_data: Form submission data with experience level
        current_keywords: List of current skills

    Returns:
        List of advanced role keywords
    """
    experience_level = candidate_data.get("ExperienceLvl", "Just starting out")

    # Map experience levels to advancement keywords
    advancement_map = {
        "Just starting out": ["Junior", "Assistant"],
        "Some experience": ["Senior", "Lead"],
        "Experienced": ["Manager", "Principal", "Head of"],
        "Very experienced": ["Director"]
    }

    advancement_terms = advancement_map.get(experience_level, ["Senior"])

    # Generate future role keywords
    future_keywords = []
    primary_skill = current_keywords[0] if current_keywords else "Professional"

    for term in advancement_terms[:2]:
        future_keywords.append(f"{term} {primary_skill}")

    return future_keywords


async def search_adzuna_jobs(
    keywords: List[str],
    location: str = "london",
    results_per_page: int = 5,
    sort_by: str = "relevance",
    page: int = 1
) -> Optional[List[Dict[str, Any]]]:
    """
    Search for jobs on Adzuna API

    Args:
        keywords: List of skills/keywords to search for
        location: Location to search in (default: "london")
        results_per_page: Number of results to return (1-50)
        sort_by: Sort order ("relevance" or "date")
        page: Page number for pagination

    Returns:
        List of job dictionaries or None if API call fails
    """
    try:
        app_id = os.environ.get("ADZUNA_APP_ID")
        app_key = os.environ.get("ADZUNA_APP_KEY")

        if not app_id or not app_key:
            print("Adzuna credentials not configured")
            return None

        # Construct API URL
        base_url = f"https://api.adzuna.com/v1/api/jobs/gb/search/{page}"

        # Build query parameters
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": min(results_per_page, 50),  # Max 30
            "what": " ".join(keywords),
            "where": location,
            "sort_by": sort_by
        }

        print(f"Adzuna search: what='{params['what']}', where='{location}'")

        # Make API request
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            print(f"Adzuna API returned {len(results)} jobs")
            return results

    except httpx.TimeoutException:
        print(f"Adzuna API timeout for query: {keywords}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Adzuna API HTTP error {e.response.status_code}: {keywords}")
        return None
    except Exception as e:
        print(f"Adzuna API error: {str(e)}")
        return None

def deduplicate_jobs(jobs: List[Dict]) -> List[Dict]:
    """
    Remove duplicate jobs based on company + title

    Args:
        jobs: List of job dictionaries from Adzuna

    Returns:
        Deduplicated list of jobs
    """
    seen_companies = set()
    unique_jobs = []

    for job in jobs:
        company = job.get("company", {}).get("display_name", "Unknown").lower().strip()
        title = job.get("title", "").lower().strip()

        # Makes sure its not the same company
        if company not in seen_companies and title and company:
            seen_companies.add(company)
            unique_jobs.append(job)

    return unique_jobs


def format_job_for_response(adzuna_job: Dict, job_type: str) -> Dict:
    """
    Extract and format relevant fields from Adzuna job

    Args:
        adzuna_job: Raw job dict from Adzuna API
        job_type: "current" or "future"

    Returns:
        Formatted job dictionary
    """
    # Format salary
    salary_min = adzuna_job.get("salary_min")
    salary_max = adzuna_job.get("salary_max")

    if salary_min and salary_max and (salary_min != salary_max):
        salary = f"£{salary_min:,.0f} - £{salary_max:,.0f}"
    elif salary_min:
        salary = f"£{salary_min:,.0f}+"
    elif salary_max:
        salary = f"Up to £{salary_max:,.0f}"
    else:
        salary = "Not specified"

    return {
        "job_type": job_type,
        "title": adzuna_job.get("title", "N/A"),
        "company": adzuna_job.get("company", {}).get("display_name", "N/A"),
        "location": adzuna_job.get("location", {}).get("display_name", "N/A"),
        "description": adzuna_job.get("description", "")[:500],  # Truncate
        "salary": salary,
        "contract_time": adzuna_job.get("contract_time", "N/A"),
        "url": adzuna_job.get("redirect_url", ""),
        "posted_date": adzuna_job.get("created", "N/A"),
    }


async def get_job_recommendations(
    candidate_data: Dict[str, Any],
    cv_analysis: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get job recommendations for a candidate (8 current + 2 future jobs)
    
    5 from London, 5 from rest of UK

    Args:
        candidate_data: Form submission data with skills and experience
        cv_analysis: OpenAI CV analysis results (can be None)

    Returns:
        List of 10 job recommendations with job_type field
    """
    try:
        # Extract keywords for current and future jobs
        current_keywords = extract_current_keywords(candidate_data, cv_analysis)
        future_keywords = extract_future_keywords(candidate_data, current_keywords)

        if not current_keywords:
            print("No keywords found for job search")
            return []

        print(f"Job search keywords - Current: {current_keywords}, Future: {future_keywords}")

        # Search for current jobs (target 10-12 to allow for deduplication)
        london_jobs_raw = []
        uk_jobs_raw = []
        current_jobs = []
        future_jobs = []
        
        target_cities = ["birmingham", "glasgow", "manchester", "leeds", "liverpool"]

        # Make 4 searches for current London
        for keyword in current_keywords[:5]:
            jobs = await search_adzuna_jobs(
                keywords=[keyword],
                location="london",
                results_per_page=10,
                sort_by="relevance"
            )
            if jobs:
                london_jobs_raw.extend(jobs)
                
        # Add top 4 unique London jobs
        current_jobs.extend(deduplicate_jobs(london_jobs_raw)[:4])
        
        # Making 4 Searches for current UK 
        for keyword in current_keywords[:5]:
            for city in target_cities: 
                jobs = await search_adzuna_jobs(
                    keywords=[keyword],
                    location=city,
                    results_per_page=5,
                    sort_by="relevance"
                )
                if jobs:
                    uk_jobs_raw.extend(jobs)
                
        # Filter: Remove any job where location contains "London"
        non_london_jobs = []
        for job in deduplicate_jobs(uk_jobs_raw):
            loc_name = job.get('location', {}).get('display_name', '').lower()
            print(f"Loc Name for Current UK jobs: {loc_name}")
            if "london" not in loc_name:  # <--- THIS IS THE FIX
                non_london_jobs.append(job)
                
        # Add top 4 unique UK jobs 
        current_jobs.extend(deduplicate_jobs(non_london_jobs)[:4])
            
        # Search for future jobs
        future_jobs_london = []
        future_jobs_uk = []

        # London based
        for keyword in future_keywords[:1]:
            jobs = await search_adzuna_jobs(
                keywords=[keyword],
                location="london",
                results_per_page=5,
                sort_by="relevance"
            )
            if jobs:
                future_jobs_london.extend(jobs)
                
        future_jobs.extend(deduplicate_jobs(future_jobs_london)[:1])
                
        # Uk based        
        for keyword in future_keywords[:1]:
            for city in target_cities: 
                jobs = await search_adzuna_jobs(
                    keywords=[keyword],
                    location=city,
                    results_per_page=5,
                    sort_by="relevance"
                )
            if jobs:
                future_jobs_uk.extend(jobs)
                
        # Filter: Remove any job where location contains "London"
        non_future_ldn_jobs = []
        for job in deduplicate_jobs(future_jobs_uk):
            loc_name = job.get('location', {}).get('display_name', '').lower()
            if "london" not in loc_name:  # <--- THIS IS THE FIX
                non_future_ldn_jobs.append(job)
                
        future_jobs.extend(deduplicate_jobs(non_future_ldn_jobs)[:1])

        # Format for response
        formatted_jobs = []

        for job in current_jobs:
            formatted_jobs.append(format_job_for_response(job, "current"))

        for job in future_jobs:
            formatted_jobs.append(format_job_for_response(job, "future"))

        print(f"Job recommendations prepared: {len(current_jobs)} current, {len(future_jobs)} future")
        print(f"Keywords for current:{current_keywords[:5]}, Future keywords:{future_keywords[:2]}")

        return formatted_jobs

    except Exception as e:
        print(f"Error getting job recommendations: {str(e)}")
        return []


# ============================================================================
# TOP SKILLS CORPORATE TRANSLATION FUNCTIONS (Phase 4)
# ============================================================================

def extract_top_skills_for_translation(
    candidate_data: Dict[str, Any],
    cv_analysis: Optional[Dict[str, Any]],
    cv_text: Optional[str]
) -> List[str]:
    """
    Extract top 3 skills from candidate data for corporate translation

    Prioritizes:
    1. Skills with concrete examples in CV
    2. Technical skills from CV analysis
    3. Self-reported skills from form

    Args:
        candidate_data: Form submission data
        cv_analysis: OpenAI CV analysis results
        cv_text: Raw CV text

    Returns:
        List of 3 skill descriptions with context
    """
    skills_with_scores = []

    # Priority 1: Extract skills from CV work experience with context
    if cv_analysis and "work_experience" in cv_analysis:
        work_exp = cv_analysis.get("work_experience", {})
        roles = work_exp.get("roles", [])

        # Add 1 role/responsibilities as skills with context
        for role in roles[:1]:  
            skills_with_scores.append({
                "description": role,
                "score": 3,
                "source": "cv_experience"
            })
            
    # Priority 2: Technical skills from CV analysis
    if cv_analysis and "skills" in cv_analysis:
        tech_skills = cv_analysis.get("skills", {}).get("technical", [])
        for skill in tech_skills[:2]:
            # Check if skill already added (avoid duplicates)
            if not any(skill.lower() in s["description"].lower() for s in skills_with_scores):
                skills_with_scores.append({
                    "description": skill,
                    "score": 2,
                    "source": "cv_technical"
                })

    # Priority 3: BasicSkills from form
    basic_skills = candidate_data.get("BasicSkills", [])
    for skill in basic_skills[:3]:
        if not any(skill.lower() in s["description"].lower() for s in skills_with_scores):
            skills_with_scores.append({
                "description": skill,
                "score": 1,
                "source": "form_basic"
            })


    # Priority 4: OtherSkills from form (comma-separated)
    other_skills_str = candidate_data.get("OtherSkills", "")
    if other_skills_str and other_skills_str.strip():
        other_skills = [s.strip() for s in other_skills_str.split(",") if s.strip()]
        for skill in other_skills[:3]:
            if not any(skill.lower() in s["description"].lower() for s in skills_with_scores):
                skills_with_scores.append({
                    "description": skill,
                    "score": 1,
                    "source": "form_other"
                })

    # Bonus: Skills mentioned multiple times get +1 score

    # Sort by score (highest first) and select top 3
    skills_with_scores.sort(key=lambda x: x["score"], reverse=True)
    top_skills = skills_with_scores[:3]

    # Return just the descriptions
    skill_descriptions = [s["description"] for s in top_skills]

    print(f"Extracted {len(skill_descriptions)} top skills for translation: {skill_descriptions}")

    return skill_descriptions


async def translate_skills_to_corporate(skills_to_translate: List[str]) -> List[Dict[str, str]]:
    """
    Translate casual/student skills into professional corporate terminology using OpenAI

    Args:
        skills_to_translate: List of 1-3 skill descriptions in casual language

    Returns:
        List of dicts with 'original', 'corporate', and 'category' fields
    """
    try:
        if not skills_to_translate or len(skills_to_translate) == 0:
            return []

        # Get OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not configured for skills translation")
            # Return original skills with default category
            return [
                {
                    "original": skill,
                    "corporate": skill,
                    "category": "ex. professional, technical, corporate"
                }
                for skill in skills_to_translate
            ]

        # Prepare the prompt
        system_prompt = """You are a professional resume writer specializing in translating casual or student experience into corporate/professional terminology.

Your task: Transform casual skill descriptions into polished, industry-standard professional skills.

Guidelines:
- Use action-oriented, concrete language
- Maintain accuracy - don't exaggerate
- Use industry-standard terminology
- Keep it concise (max 6-8 words)
- Categorize as: technical, leadership, professional, analytical, creative

Examples:
Input: "Team leader in university projects"
Output: "Project Management & Team Leadership" (category: leadership)

Input: "Organised charity events"
Output: "Event Coordination & Cross-functional Collaboration" (category: professional)

Input: "Good with Excel"
Output: "Data Analysis & Financial Modeling" (category: technical)

Input: "Python programming"
Output: "Python Development & Programming" (category: technical)"""

        # Build numbered list of skills
        skills_list = "\n".join([f"{i+1}. {skill}" for i, skill in enumerate(skills_to_translate)])

        user_prompt = f"""Transform these {len(skills_to_translate)} skill(s) into professional corporate terminology:

{skills_list}

Return as JSON array with this exact structure:
[
  {{"original": "...", "corporate": "...", "category": "..."}},
  {{"original": "...", "corporate": "...", "category": "..."}}
]

Ensure you return exactly {len(skills_to_translate)} item(s) in the array."""

        print(f"Translating {len(skills_to_translate)} skills to corporate terminology...")

        # Call OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)

        # Handle different possible response formats
        if isinstance(result, dict) and "skills" in result:
            translated_skills = result["skills"]
        elif isinstance(result, dict) and "translations" in result:
            translated_skills = result["translations"]
        elif isinstance(result, list):
            translated_skills = result
        else:
            # Try to extract array from dict
            for key, value in result.items():
                if isinstance(value, list):
                    translated_skills = value
                    break
            else:
                raise ValueError("Could not find skills array in response")

        # Validate structure
        for skill in translated_skills:
            if "original" not in skill or "corporate" not in skill or "category" not in skill:
                raise ValueError("Invalid skill structure in response")

        print(f"Successfully translated {len(translated_skills)} skills to corporate terminology")

        return translated_skills

    except Exception as e:
        print(f"Error translating skills to corporate terminology: {str(e)}")

        # Fallback: Return original skills with default categories
        return [
            {
                "original": skill,
                "corporate": skill,
                "category": "professional"
            }
            for skill in skills_to_translate
        ]

"""
async def send_to_webflow_cms(
    submission_id: str,
    analysis_data: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    
    #Send analysis results to Webflow CMS to create a results page

    #Args:
        #submission_id: Unique submission ID
        #analysis_data: Complete analysis data from webhook

    #Returns:
        #Dict with webflow_item_id and results_url, or None if failed
    
    try:
        api_token = os.environ.get("WEBFLOW_API_TOKEN")
        collection_id = os.environ.get("WEBFLOW_COLLECTION_ID")

        if not api_token or not collection_id:
            print("Webflow credentials not configured")
            return None

        # Extract data
        candidate = analysis_data.get("candidate", {})
        score = analysis_data.get("Employability Score", {})
        skills = analysis_data.get("top_skills_corporate", [])
        jobs = analysis_data.get("Suggested roles", [])
        cv_analysis = analysis_data.get("CV Analysis", [])

        # Format skills
        skill_texts = [
            f"{s.get('corporate', s.get('original', 'N/A'))}"
            for s in skills[:3]
        ]
        # Pad with empty strings if less than 3 skills
        while len(skill_texts) < 3:
            skill_texts.append("")

        # Format score breakdown
        score_breakdown = (
            f"CV Quality: {score.get('cv_quality', 0)}/30, "
            f"Skills: {score.get('skills_match', 0)}/25, "
            f"Experience: {score.get('experience', 0)}/25, "
            f"Personality: {score.get('personality_fit', 0)}/20"
        )

        # Format job recommendations (top 5 for display)
        job_summary = "\n".join([
            f"- {job.get('title', 'N/A')} at {job.get('company', 'N/A')}"
            for job in jobs[:5]
        ])

        # Create slug (URL-safe identifier)
        slug = f"{submission_id}-{datetime.utcnow().strftime('%Y%m%d')}"

        # Prepare request
        url = f"https://api.webflow.com/v2/collections/{collection_id}/items"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        body = {
            "isArchived": False,
            "isDraft": False,
            "fieldData": {
                "slug": slug,
                "name": candidate.get("name", "Unknown"),
                "email": candidate.get("email", ""),
                "employability-score": str(score.get("total", 0)),
                #"score-breakdown": score_breakdown,
                #"top-skill-1": skill_texts[0],
                #"top-skill-2": skill_texts[1],
                #"top-skill-3": skill_texts[2],
                "suggested-roles-2": job_summary,
                #"cv-analysis": cv_analysis
                #"submission-timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }

        # Make request
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()

            result = response.json()
            webflow_id = result.get("id")

            # Construct results URL (***Need to change to dynamic)
            results_url = f"www.ukngn.com/form-results/applicant-006"

            print(f"Successfully sent to Webflow CMS: {webflow_id}")

            return {
                "webflow_item_id": webflow_id,
                "results_url": results_url
            }

    except httpx.TimeoutException:
        print("Webflow API timeout")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Webflow API HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error sending to Webflow CMS: {str(e)}")
        return None
"""
async def send_to_webflow_cms(
    submission_id: str,
    analysis_data: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    Sends results to Webflow CMS.
    Updated to read 'Employability Score' and 'Suggested roles' keys.
    """
    try:
        api_token = os.environ.get("WEBFLOW_API_TOKEN")
        collection_id = os.environ.get("WEBFLOW_COLLECTION_ID")

        if not api_token or not collection_id:
            print("Webflow credentials not configured")
            return None

        # 1. Extract Data using YOUR CUSTOM KEYS
        candidate = analysis_data.get("candidate", {})
        
        # KEY 1: "Employability Score" (Dict with 'total')
        score_data = analysis_data.get("Employability Score", {}) 
        
        # KEY 2: "Suggested roles" (List of jobs)
        jobs = analysis_data.get("Suggested roles", []) 
        
        # KEY 3: Skills
        skills = analysis_data.get("top_skills_corporate", []) 

        # Format skills safely
        skill_texts = [
            f"{s.get('corporate', s.get('original', 'N/A'))}"
            for s in skills[:3]
        ]
        while len(skill_texts) < 3: skill_texts.append("")

        # Format job recommendations
        job_summary = "\n".join([
            f"- {job.get('title', 'N/A')} at {job.get('company', 'N/A')}"
            for job in jobs[:5]
        ])

        slug = f"{submission_id}"
        url = f"https://api.webflow.com/v2/collections/{collection_id}/items"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        # 2. Map to Webflow CMS Slugs
        # IMPORTANT: Replace these keys with the ACTUAL SLUGS from your check_slugs.py script
        body = {
            "isArchived": False,
            "isDraft": False,
            "fieldData": {
                "name": candidate.get("name", "Unknown"),
                "slug": slug,
                "candidate-email": candidate.get("email", ""),
                
                # Mapping 'Employability Score' -> 'employability-score'
                "employability-score": str(score_data.get("total", 0)), 
                
                # Mapping 'Suggested roles' -> 'job-recommendations-2' (or whatever your slug is)
                "job-recommendations-2": job_summary, 
                
                "top-skill-1": skill_texts[0],
                "top-skill-2": skill_texts[1],
                "top-skill-3": skill_texts[2],
                "submission-timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Use just the ID for the URL since we made slug = id
            results_url = f"www.ukngn.com/form-results/applicant-006"
            #f"/form-results/form-result-page/{slug}" 
            
            print(f"Successfully sent to Webflow CMS: {result.get('id')}")
            return {
                "webflow_item_id": result.get("id"),
                "results_url": results_url
            }

    except Exception as e:
        print(f"Error sending to Webflow CMS: {str(e)}")
        # Check for 400 errors specifically to help debug slug issues
        if "response" in locals() and response.status_code == 400:
             print(f"Webflow Validation Error: {response.text}")
        return None


def save_analysis_to_json(submission_id: str, analysis_data: Dict[str, Any]) -> bool:
    """
    Log analysis to console (Persistent) and try to save to file (Ephemeral)
    """
    try:
        # 1. ALWAYS Log to console (This is your persistent storage for now)
        print(f"--- COMPLETE ANALYSIS FOR {submission_id} ---")
        print(json.dumps(analysis_data, indent=2, default=str, ensure_ascii=False))
        print(f"---------------------------------------------")

        # 2. Try to save to file (existing logic)
        if os.environ.get("VERCEL"):
            results_dir = "/tmp/analysis_results"
        else:
            results_dir = "analysis_results"

        os.makedirs(results_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/{submission_id}_{timestamp}.json"

        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str, ensure_ascii=False)

        print(f"Analysis saved to {filename}")
        return True

    except Exception as e:
        print(f"Error saving analysis to JSON: {str(e)}")
        return False


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

    # Needs to be  changed *** THiss cant be count-based
    breakdown["skills_match"] = min(25, skills_count * 3)

    # 3. Experience Level (0-25 points)
    experience_mapping = {
        "Just starting out": 6,
        "Some experience": 12,
        "Experienced": 18,
        "Very experienced": 25
    }
    experience_level = form_data.get("ExperienceLvl", "Just starting out")
    breakdown["experience"] = experience_mapping.get(experience_level, 10)

    # 4. Personality Fit (0-20 points)
    people_score = form_data.get("People", 3)
    structured_score = form_data.get("StructuredTask", 3)
    initiative_score = form_data.get("InitiativeTask", 3)
    breakdown["personality_fit"] = int(((people_score + structured_score + initiative_score) / 15) * 20)

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
    
import os
import glob

@app.get("/debug-tmp")
async def list_temp_files():
    """
    Lists files in /tmp and reads the content of JSON/text files.
    """
    import glob
    
    # Get all files
    files = glob.glob("/tmp/**/*", recursive=True)
    file_details = []

    for f in files:
        # Only process if it is a file (not a directory or socket)
        if os.path.isfile(f):
            try:
                size = os.path.getsize(f)
                content = "[Binary or Large File - Not Read]"
                
                # If it's a JSON/Text file and small (< 1MB), read it
                if size < 1_000_000 and (f.endswith(".json") or f.endswith(".txt") or f.endswith(".log")):
                    with open(f, "r", encoding="utf-8", errors="replace") as file_obj:
                        content = file_obj.read()
                        
                        # If it's JSON, try to parse it so it looks pretty
                        try:
                            content = json.loads(content)
                        except:
                            pass # Keep as string if parsing fails

                file_details.append({
                    "path": f, 
                    "size_bytes": size,
                    "content": content
                })
            except Exception as e:
                file_details.append({"path": f, "error": str(e)})

    return {
        "count": len(file_details),
        "files": file_details
    }
"""
# New function for webflow since it requires an addtional key 
@app.post("/webhook/webflow")
async def receive_webflow_webhook(request: Request):
    
    #Main webhook endpoint to receive Webflow Form submissions
    
    start_time = time.time()
    
    try:
        # 1. Get Raw Data (Required for verification)
        body_bytes = await request.body()
        headers = dict(request.headers)
        
        # 2. SECURITY CHECK: Verify Signature
        secret = os.environ.get("WEBFLOW_WEBHOOK_SECRET")
        if secret:
            is_valid = verify_webflow_signature(body_bytes, headers, secret)
            if not is_valid:
                print("Security Alert: Invalid Webflow Signature")
                raise HTTPException(status_code=401, detail="Invalid Signature")
        else:
            print("WARNING: WEBFLOW_WEBHOOK_SECRET not set. Skipping check.")

        # 3. Parse JSON
        try:
            full_payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})

        # 4. Extract Core Data (Handle Webflow Structure)
        # Webflow wraps the real data inside "payload" -> "data"
        payload_root = full_payload.get("payload", {})
        form_data = payload_root.get("data", {})
        
        # Get IDs
        submission_id = payload_root.get("id") or full_payload.get("siteId") # Use Webflow's ID
        
        # Log the received data
        log_webhook_data("webflow_submission", form_data, headers)

        # 5. Map Webflow Fields to Your Logic
        # NOTE: You MUST check your Webflow Form field names! 
        # They might be "Name" instead of "Full Name". Adjust keys below:
        mapped_data = {
            "FullName": form_data.get("FullName","Unknown"),
            "Email": form_data.get("Email") or form_data.get("email"),
            "DoB": form_data.get("DoB"),
            "PhoneNo": form_data.get("PhoneNo") or form_data.get("Phone Number"),
            # Map skills manually or join all text fields if unsure
            "BasicSkills": [form_data.get("BasicSkills", "")] if form_data.get("Skills") else [],
            "OtherSkills": form_data.get("OtherSkills", ""),
            "ExperienceLvl": form_data.get("ExperienceLvl", "Just starting out"),
            # Map Score fields (Ensure your Webflow form sends these as numbers)
            "People": int(form_data.get("People", 3)),
            "StructuredTask": int(form_data.get("StructuredTask", 3)),
            "InitiativeTask": int(form_data.get("Initiative Score", 3))
        }

        # Initialize Response
        response_data = {
            "status": "success",
            "submission_id": submission_id,
            "candidate": {"name": mapped_data["FullName"], "email": mapped_data["Email"]},
            "Employability Score": 0,
            "Suggested roles": [],
            "CV Analysis": [],
            "errors": []
        }

        # 6. CV Processing (Handle Webflow File Uploads)
        cv_url = None
        # Webflow sends file uploads as a direct URL string in the data key
        # Check keys like "CV", "Resume", "File"
        for key, value in form_data.items():
            if "cv" in key.lower() or "resume" in key.lower() or "upload" in key.lower():
                if isinstance(value, str) and value.startswith("http"):
                    cv_url = value
                    break
        
        cv_analysis = None
        cv_text = None
        
        if cv_url:
            print(f"Downloading CV from: {cv_url}")
            pdf_bytes = await download_pdf(cv_url)
            if pdf_bytes:
                cv_text = extract_text_from_pdf(pdf_bytes)
                if cv_text:
                    # Pass mapped_data instead of raw payload
                    cv_analysis = await analyze_cv_with_openai(cv_text, mapped_data)

        # 7. Calculate Score & Get Jobs
        # Pass mapped_data because it has the clean keys your functions expect
        employability_score = calculate_employability_score(cv_analysis, mapped_data)
        job_recommendations = await get_job_recommendations(mapped_data, cv_analysis)
        
        # 8. Send Results to CMS (Same as before)
        response_data["Employability Score"] = employability_score
        response_data["Suggested roles"] = job_recommendations
        response_data["CV Analysis"] = cv_analysis
        
        
        # ... (Include your Top Skills Translation logic here if needed) ...

        webflow_result = await send_to_webflow_cms(submission_id, response_data)
        
        # Final Success Response
        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        print(f"Error processing Webflow webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
"""

@app.post("/webhook/webflow")
async def receive_webflow_webhook(request: Request):
    """
    Main webhook endpoint to receive Webflow Form submissions
    """
    start_time = time.time()
    
    try:
        # 1. Get Raw Data (Required for verification)
        body_bytes = await request.body()
        headers = dict(request.headers)
        
        # 2. SECURITY CHECK: Verify Signature
        secret = os.environ.get("WEBFLOW_WEBHOOK_SECRET")
        if secret:
            is_valid = verify_webflow_signature(body_bytes, headers, secret)
            if not is_valid:
                print("Security Alert: Invalid Webflow Signature")
                raise HTTPException(status_code=401, detail="Invalid Signature")
        else:
            print("WARNING: WEBFLOW_WEBHOOK_SECRET not set. Skipping check.")

        # 3. Parse JSON
        try:
            full_payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})

        # 4. Extract Core Data
        # Webflow wraps data in "payload" -> "data"
        payload_root = full_payload.get("payload", {})
        form_data = payload_root.get("data", {})
        submission_id = payload_root.get("id") or full_payload.get("siteId")
        
        # Log the received data
        log_webhook_data("webflow_submission", form_data, headers)

        # 5. Map Webflow Fields to Internal Logic
        # Update these keys to match YOUR Webflow form's exact field names
        mapped_data = {
            "FullName": form_data.get("Name") or form_data.get("Full Name") or form_data.get("name"),
            "Email": form_data.get("Email") or form_data.get("email"),
            "DoB": form_data.get("Date of Birth") or form_data.get("DoB"),
            "PhoneNo": form_data.get("Phone") or form_data.get("Phone Number"),
            # Skills often come as a single string or comma-separated list
            "BasicSkills": [s.strip() for s in form_data.get("Skills", "").split(',')] if form_data.get("Skills") else [],
            "OtherSkills": form_data.get("Other Skills", ""),
            "ExperienceLvl": form_data.get("Experience Level", "Just starting out"),
            # Ensure scores are integers (default to 3 if missing)
            "People": int(form_data.get("People Score") or 3),
            "StructuredTask": int(form_data.get("Structure Score") or 3),
            "InitiativeTask": int(form_data.get("Initiative Score") or 3)
        }

        # Initialize Response with your PREFERRED keys
        response_data = {
            "status": "success",
            "submission_id": submission_id,
            "candidate": {"name": mapped_data["FullName"], "email": mapped_data["Email"]},
            "Employability Score": None,  # Capitalized as requested
            "Suggested roles": [],        # Capitalized as requested
            "CV Analysis": None,          # Capitalized as requested
            "top_skills_corporate": [],
            "errors": []
        }

        # 6. CV Processing
        cv_url = None
        for key, value in form_data.items():
            if "cv" in key.lower() or "resume" in key.lower() or "upload" in key.lower():
                if isinstance(value, str) and value.startswith("http"):
                    cv_url = value
                    break
        
        cv_analysis = None
        cv_text = None
        
        if cv_url:
            print(f"Downloading CV from: {cv_url}")
            # FIX: Added follow_redirects=True to handle Webflow 302 redirects
            pdf_bytes = await download_pdf(cv_url) 
            if pdf_bytes:
                cv_text = extract_text_from_pdf(pdf_bytes)
                if cv_text:
                    cv_analysis = await analyze_cv_with_openai(cv_text, mapped_data)

        # 7. Calculate Score & Get Jobs
        employability_score = calculate_employability_score(cv_analysis, mapped_data)
        job_recommendations = await get_job_recommendations(mapped_data, cv_analysis)
        
        # 8. Assign Data to Response
        response_data["Employability Score"] = employability_score
        response_data["Suggested roles"] = job_recommendations
        response_data["CV Analysis"] = cv_analysis

        # 9. Top Skills Translation (Restored logic)
        if cv_analysis and cv_text:
            top_skills_raw = extract_top_skills_for_translation(mapped_data, cv_analysis, cv_text)
            if top_skills_raw:
                top_skills_corporate = await translate_skills_to_corporate(top_skills_raw)
                response_data["top_skills_corporate"] = top_skills_corporate

        # 10. Send to Webflow CMS
        # This function handles the slug mapping
        webflow_result = await send_to_webflow_cms(submission_id, response_data)
        
        if webflow_result:
            response_data["webflow_results_url"] = webflow_result.get("results_url")

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        print(f"Error processing Webflow webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/webhook/fillout")
async def receive_fillout_webhook(request: Request):
    """
    Main webhook endpoint to receive Fillout.com submissions

    Phase 2 Implementation:
    1. Receives the webhook POST request
    2. Downloads CV from Fillout S3
    3. Extracts text from PDF
    4. Analyzes CV with OpenAI GPT-4o
    5. Calculates employability score (0-100)
    6. Returns comprehensive results

    Future enhancements:
    - Match jobs with Adzuna (Phase 3)
    - Send results to Webflow (Phase 4)
    """
    start_time = time.time()

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
        for key in ["SubmissionID", "submissionId", "submission_id", "id", "Submission ID"]:
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

        # Parse payload with new model
        try:
            webhook_data = FilloutWebhookPayload(**payload)
        except Exception as validation_error:
            print(f"Payload validation warning: {str(validation_error)}")
            # Continue with raw payload if validation fails
            webhook_data = None

        # Initialize response structure
        response_data = {
            "status": "success",
            "submission_id": submission_id,
            "candidate": {
                "name": payload.get("FullName", "Unknown"),
                "email": candidate_email
            },
            "employability_score": None,
            "cv_analysis": None,
            "recommendations": {
                "next_steps": [],
                "suggested_roles": []
            },
            "top_skills_corporate": [],
            "webflow_results_url": None,
            "webflow_item_id": None,
            "processing_time_ms": 0,
            "errors": []
        }

        # Phase 2: CV Processing
        cv_analysis = None
        cv_url = None
        cv_text = None  
        # Extract CV URL
        if "CV" in payload and isinstance(payload["CV"], list) and len(payload["CV"]) > 0:
            cv_url = payload["CV"][0].get("url")

        if cv_url:
            print(f"Processing CV from URL: {cv_url[:50]}...")

            # Step 1: Download PDF
            pdf_bytes = await download_pdf(cv_url)

            if pdf_bytes:
                # Step 2: Extract text
                cv_text = extract_text_from_pdf(pdf_bytes)

                if cv_text:
                    # Step 3: Analyze with OpenAI
                    cv_analysis = await analyze_cv_with_openai(cv_text, payload)

                    if cv_analysis:
                        response_data["cv_analysis"] = cv_analysis
                    else:
                        response_data["errors"].append("OpenAI analysis failed")
                else:
                    response_data["errors"].append("PDF text extraction failed")
            else:
                response_data["errors"].append("PDF download failed")
        else:
            response_data["errors"].append("No CV URL found in payload")

        # Step 4: Calculate employability score
        employability_score = calculate_employability_score(cv_analysis, payload)
        response_data["employability_score"] = employability_score

        # Phase 3: Job Recommendations via Adzuna
        job_recommendations = await get_job_recommendations(payload, cv_analysis)

        if job_recommendations:
            response_data["recommendations"]["suggested_roles"] = job_recommendations
            print(f"Added {len(job_recommendations)} job recommendations to response")
        else:
            response_data["recommendations"]["suggested_roles"] = []
            if not os.environ.get("ADZUNA_APP_ID") or not os.environ.get("ADZUNA_APP_KEY"):
                response_data["recommendations"]["suggested_roles"] = [{
                    "message": "Job recommendations require Adzuna API credentials",
                    "setup_instructions": "Visit https://developer.adzuna.com/ to register"
                }]
            else:
                response_data["errors"].append("Job recommendations temporarily unavailable")

        # Phase 4: Top Skills Corporate Translation
        if cv_analysis and cv_text:
            top_skills_raw = extract_top_skills_for_translation(payload, cv_analysis, cv_text)

            if top_skills_raw and len(top_skills_raw) > 0:
                top_skills_corporate = await translate_skills_to_corporate(top_skills_raw)
                response_data["top_skills_corporate"] = top_skills_corporate
                print(f"Added {len(top_skills_corporate)} corporate skill translations to response")
            else:
                print("No skills found for corporate translation")
        else:
            # Fallback: Extract from form data only if no CV
            if not cv_analysis:
                basic_skills = payload.get("BasicSkills", [])
                other_skills = payload.get("OtherSkills", "")

                fallback_skills = []
                fallback_skills.extend(basic_skills[:2])

                if other_skills and other_skills.strip():
                    other_skills_list = [s.strip() for s in other_skills.split(",") if s.strip()]
                    fallback_skills.extend(other_skills_list[:1])

                if fallback_skills:
                    fallback_skills = fallback_skills[:3]
                    top_skills_corporate = await translate_skills_to_corporate(fallback_skills)
                    response_data["top_skills_corporate"] = top_skills_corporate
                    print(f"Added {len(top_skills_corporate)} corporate skill translations from form data")

        # Add recommendations (Static next_steps) -> CHanged to dynamic later
        if employability_score["total"] >= 70:
            response_data["recommendations"]["next_steps"] = [
                "Your profile is strong! Focus on networking and applying to target companies.",
                "Consider specialized certifications to stand out further.",
                "Practice interview skills to maximize your opportunities."
            ]
        elif employability_score["total"] >= 50:
            response_data["recommendations"]["next_steps"] = [
                "Build more practical experience through projects or internships.",
                "Expand your skill set in your target industry.",
                "Enhance your CV with quantifiable achievements."
            ]
        else:
            response_data["recommendations"]["next_steps"] = [
                "Focus on building foundational skills in your chosen field.",
                "Consider entry-level positions or apprenticeships to gain experience.",
                "Invest in relevant courses or certifications.",
                "Seek mentorship from professionals in your target industry."
            ]

        # Phase 5: Send to Webflow CMS
        webflow_result = await send_to_webflow_cms(submission_id, response_data)

        if webflow_result:
            response_data["webflow_results_url"] = webflow_result["results_url"]
            response_data["webflow_item_id"] = webflow_result["webflow_item_id"]
            print(f"Results page created: {webflow_result['results_url']}")
        else:
            print("Webflow CMS integration skipped or failed")

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        response_data["processing_time_ms"] = processing_time

        # Save complete analysis to JSON file
        complete_analysis = {
            "submission_id": submission_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "candidate": response_data["candidate"],
            "form_data": {
                "basic_skills": payload.get("BasicSkills", []),
                "other_skills": payload.get("OtherSkills", ""),
                "experience_level": payload.get("ExperienceLvl", ""),
                "soft_skills": payload.get("SoftSkills", []),
                "people_score": payload.get("People", 0),
                "structured_task_score": payload.get("StructuredTask", 0),
                "initiative_score": payload.get("InitiativeTask", 0),
                "linkedin": payload.get("Linkedin", ""),
                "phone": payload.get("PhoneNo", ""),
                "dob": payload.get("DoB", "")
            },
            "cv_analysis": response_data["cv_analysis"],
            "employability_score": response_data["employability_score"],
            "recommendations": response_data["recommendations"],
            "top_skills_corporate": response_data["top_skills_corporate"],
            "webflow_results_url": response_data["webflow_results_url"],
            "webflow_item_id": response_data["webflow_item_id"],
            "processing_time_ms": processing_time,
            "errors": response_data["errors"]
        }

        save_analysis_to_json(submission_id, complete_analysis)

        print(f"Webhook processed successfully - Submission: {submission_id}, Score: {employability_score['total']}/100, Time: {processing_time}ms")

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        # Log error but still return 200 to webhook provider
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

        # Return error response with basic score
        processing_time = int((time.time() - start_time) * 1000)

        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "message": "Error occurred during processing",
                "error": str(e),
                "employability_score": {
                    "total": 0,
                    "breakdown": {
                        "cv_quality": 0,
                        "skills_match": 0,
                        "experience": 0,
                        "personality_fit": 0
                    },
                    "grade": "N/A",
                    "percentile": 0
                },
                "processing_time_ms": processing_time
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

