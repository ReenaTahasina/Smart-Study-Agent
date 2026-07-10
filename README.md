# Smart Study Agent

An AI-powered study companion built with **IBM Granite-4** via **watsonx.ai**.

## Features

| Feature | Description |
|---|---|
| 📄 Summary Generator | Structured topic summaries with key concepts and takeaways |
| 📅 Study Plan | Day-by-day personalised study schedule |
| 🃏 Flashcards | Interactive flip-card Q&A sets |
| 🎯 Quiz | Multiple-choice questions with instant feedback |
| 💡 Concept Explainer | Three styles: simple, detailed, visual/analogies |
| 💬 Study Chat | Conversational AI tutor with context memory |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Configuration

Edit the `.env` file (or set environment variables):

```
WATSONX_URL=https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29
PROJECT_ID=e5338631-1ef1-4dd4-b1ae-d703b5fc87f7
MODEL_ID=ibm/granite-4-h-small
IBM_API_KEY=<your-ibm-cloud-api-key>
```

## Architecture

```
app.py               ← Flask backend + IBM Granite-4 API calls
templates/index.html ← Single-page frontend (HTML + CSS + JS)
requirements.txt     ← Python dependencies
.env                 ← API credentials (git-ignored)
```
