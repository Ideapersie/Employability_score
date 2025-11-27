# Chanci AI - CV Processing MVP

FastAPI webhook receiver deployed on Vercel Serverless that processes CV submissions from Fillout.com and generates employability assessments.

## 🎯 Project Overview

**Chanci AI** is an automated employability scoring system that:
- Receives candidate CV submissions via Fillout.com webhooks
- Processes CVs using OpenAI GPT-4o
- Matches candidates with jobs using Adzuna API
- Displays results via Webflow CMS

## 🏗️ Architecture

```
Fillout.com → Webhook → Vercel (FastAPI) → OpenAI/Adzuna → Webflow CMS
```

- **Input**: Fillout.com (forms + PDF storage)
- **Backend**: Python FastAPI on Vercel Serverless
- **Frontend**: Webflow CMS Collections
- **APIs**: OpenAI GPT-4o, Adzuna, Webflow

## 📁 Project Structure

```
Employability_score/
├── api/
│   ├── __init__.py          # Package marker
│   └── index.py             # Main FastAPI application (webhook receiver)
├── requirements.txt         # Python dependencies
├── vercel.json              # Vercel deployment configuration
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🚀 Current Status: Phase 1 - Webhook Testing

**Goal**: Deploy a test version that receives and logs Fillout webhooks

**What's Working**:
- ✅ FastAPI application with webhook endpoint
- ✅ Comprehensive logging for debugging
- ✅ Flexible payload parsing (discovers Fillout structure)
- ✅ Environment variable configuration
- ✅ Error handling and monitoring

**What's Next**:
- Configure Fillout webhook URL
- Test with real form submissions
- Analyze payload structure
- Implement CV processing with OpenAI

## 🛠️ Local Development

### Prerequisites
- Python 3.12+
- Git
- Virtual environment tool

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

   # Edit .env and add your API keys
   # OPENAI_API_KEY=sk-your-actual-key-here
   ```

5. **Run the development server**
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```

6. **Access the application**
   - API: http://localhost:8000
   - Swagger Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## ☁️ Deployment (Vercel)

### Prerequisites
- GitHub account
- Vercel account
- Repository pushed to GitHub

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add FastAPI webhook receiver"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Vercel auto-detects the configuration

3. **Configure Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add `OPENAI_API_KEY` with your OpenAI API key
   - Set for "Production" environment
   - Click "Redeploy" to apply changes

4. **Get Your Deployment URL**
   - Vercel provides: `https://your-project.vercel.app`
   - Note this URL for Fillout webhook configuration

### Auto-Deployment
- Every push to `main` branch triggers automatic deployment
- View build logs in Vercel Dashboard
- Monitor function logs in real-time

## 🔗 API Endpoints

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
    "docs": "/docs"
  },
  "environment": {
    "OPENAI_API_KEY": true,
    "ADZUNA_API_KEY": false,
    "WEBFLOW_API_KEY": false
  }
}
```

### `GET /health`
Health check endpoint for monitoring

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:30:00Z",
  "service": "webhook-receiver",
  "environment_configured": {...}
}
```

### `POST /webhook/fillout`
Main webhook endpoint to receive Fillout.com submissions

**What it does**:
1. Receives webhook POST request
2. Extracts headers and body
3. Logs complete payload structure
4. Extracts CV/PDF information
5. Returns 200 OK acknowledgment

**Test with curl**:
```bash
curl -X POST https://your-project.vercel.app/webhook/fillout \
  -H "Content-Type: application/json" \
  -d '{
    "submissionId": "test-123",
    "data": {
      "Full Name:": "John Doe",
      "Email:": "john@example.com"
    }
  }'
```

**Response**:
```json
{
  "status": "success",
  "message": "Webhook received and logged",
  "timestamp": "2025-01-27T10:30:00Z",
  "submission_id": "test-123",
  "candidate_email": "john@example.com",
  "next_steps": "CV processing will be implemented in Phase 2"
}
```

## 📊 Fillout.com Integration

### Form Fields
The Fillout form captures:
- **Personal Info**: Full Name, Email, Phone, Date of Birth, LinkedIn
- **Skills**: Basic Skills, Other Skills, Soft Skills, Experience Level
- **Personality**: Working with people, Structure preference, Initiative
- **CV Upload**: PDF file

### Webhook Configuration

1. **Go to Fillout.com**
   - Open your form settings
   - Navigate to Integrations → Webhooks

2. **Add Webhook**
   - URL: `https://your-project.vercel.app/webhook/fillout`
   - Method: `POST`
   - Content-Type: `application/json`

3. **Test the Integration**
   - Use Fillout's "Send Test" feature
   - Or submit a real form
   - Check Vercel logs for webhook data

4. **Verify Logs**
   - Vercel Dashboard → Deployments → [Latest] → Functions → Logs
   - Look for "WEBHOOK EVENT: webhook_received"
   - Review complete payload structure

## 🔍 Monitoring & Debugging

### View Logs in Vercel
1. Go to Vercel Dashboard
2. Select your project
3. Click "Deployments"
4. Select the latest deployment
5. Click "Functions" tab
6. View real-time logs

### What Gets Logged
Every webhook logs:
- Timestamp
- Event type
- Request headers
- Complete payload
- Payload structure (keys, size)
- CV/PDF metadata (if found)
- Submission ID and email

### Example Log Output
```json
{
  "timestamp": "2025-01-27T10:30:00Z",
  "event_type": "webhook_received",
  "headers": {
    "content-type": "application/json",
    "user-agent": "Fillout-Webhooks/1.0"
  },
  "payload_structure": {
    "top_level_keys": ["submissionId", "data", "questions"],
    "payload_size_bytes": 2048
  },
  "payload": {...},
  "cv_metadata": {
    "found": true,
    "location": "Please upload your CV to get started",
    "type": "url",
    "value": "https://..."
  }
}
```

## 🧪 Testing

### Local Testing
```bash
# Start the server
uvicorn api.index:app --reload --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test webhook endpoint
curl -X POST http://localhost:8000/webhook/fillout \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

### Production Testing
```bash
# Test health
curl https://your-project.vercel.app/health

# Test webhook
curl -X POST https://your-project.vercel.app/webhook/fillout \
  -H "Content-Type: application/json" \
  -d '{"submissionId": "test-123", "data": {"name": "Test"}}'
```

## 🔐 Environment Variables

### Required Now
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o (get from platform.openai.com)

### Required Later
- `ADZUNA_API_KEY`: Adzuna API credentials for job matching
- `WEBFLOW_API_KEY`: Webflow API token for CMS integration

### Configuration
- **Local**: Add to `.env` file (not committed to git)
- **Vercel**: Add in Dashboard → Settings → Environment Variables

**Security Note**: Never commit `.env` files or log actual API key values!

## 🐛 Troubleshooting

### Deployment Issues

**Build fails on Vercel**
- Check build logs in Vercel Dashboard
- Verify `requirements.txt` syntax
- Ensure all dependencies are compatible

**404 Not Found**
- Verify `vercel.json` routes configuration
- Check file is at `api/index.py`
- Confirm deployment succeeded

### Runtime Issues

**Webhook not received**
- Test with curl to isolate issue
- Verify Fillout webhook URL matches exactly
- Check Fillout webhook status/logs

**No logs appearing**
- Add explicit `print()` statements in code
- Wait a few seconds for logs to appear
- Check Dashboard → Functions → View Logs

**Environment variables not working**
- Verify variables are set in Vercel Dashboard
- Redeploy after adding variables
- Check logs show: `"OPENAI_API_KEY": true`

## 📝 Development Roadmap

### ✅ Phase 1: Webhook Testing (CURRENT)
- [x] Deploy FastAPI to Vercel
- [x] Create webhook endpoint
- [x] Implement comprehensive logging
- [ ] Configure Fillout webhook
- [ ] Test with real submissions
- [ ] Document payload structure

### 🔄 Phase 2: CV Processing
- [ ] Download PDF from Fillout
- [ ] Extract text from PDF
- [ ] Parse CV with OpenAI GPT-4o
- [ ] Store parsed data
- [ ] Generate employability score

### 🔄 Phase 3: Job Matching
- [ ] Integrate Adzuna API
- [ ] Search jobs based on skills
- [ ] Calculate match scores
- [ ] Rank opportunities

### 🔄 Phase 4: Results Display
- [ ] Integrate Webflow API
- [ ] Push results to CMS
- [ ] Create user dashboard
- [ ] Email notifications

## 🤝 Contributing

This is an MVP project. To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and on Vercel
5. Submit a pull request

## 📄 License

[Add your license here]

## 📧 Contact

For questions or support, contact [your contact info]

---

**Built with**: FastAPI, Vercel, OpenAI, Fillout.com

**Version**: 1.0.0 (Webhook Testing Phase)
