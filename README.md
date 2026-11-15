🚀 Bhishmak-SLM — Privacy-First Local Multi-Model AI Workspace

Bhishmak-SLM is a local, multi-feature Small Language Model (SLM) assistant built with Python, Gradio, and Ollama. Designed for computer science students, developers, and creators, it runs local open-weights models (phi3:mini, llama3, qwen2.5-coder, and llava) directly on consumer hardware without mandatory paid cloud APIs.

✨ Key Capabilities & Architecture

🧠 Capability-Aware Multi-Model Router: Automatically detects installed Ollama models and routes tasks across general reasoning (llama3, phi3:mini), software engineering (qwen2.5-coder), vision analysis (llava), and local embeddings (nomic-embed-text).

📚 Hybrid RAG Document Vault: Supports .pdf, .docx, .txt, .md, and source code files. Combines structure-preserving chunking with Okapi BM25, bigram phrase matching, and optional local vector embeddings fused via Reciprocal Rank Fusion (RRF).

🗂️ Persistent Chat & Categorized Memory: Retains full conversation history across restarts, archives past sessions safely, and extracts durable user profile facts while filtering out transient conversational states.

🌐 Multi-Source Web & Media Research: Integrates live web search (DuckDuckGo + Instant Answer API + Wikipedia API fallback), an SSRF-protected website content reader, and a YouTube transcript analyzer.

🕸️ Live HTML5 Sandbox: Renders generated single-file HTML/CSS/JS applications inside an interactive sandboxed preview pane.

🎨 Multimodal & Voice Utilities: Includes live screen capture, visual UI/UX auditing, 16:9 HD thumbnail generation, browser microphone dictation (Web Speech API), and text-to-speech playback (gTTS).

🛠️ 16 Specialized AI Modes

Mode Name

Category

Primary Use Case

Casual Chat & Assistant

General

Everyday Q&A, Hinglish/Hindi/English conversation, and task planning

BCA Exam Mode

Academic

University exam answers structured by marks (definitions, points, diagrams)

BCA Mock Test Engine

Academic

Generates Section A/B/C question papers and grades submitted student answers

Deep Coding Tutor Mode

Engineering

Root-cause debugging, runnable code, edge cases, and complexity analysis

Web-Dev Sandbox & Auto-Healer

Engineering

Self-contained responsive HTML5/CSS/JS apps with live preview rendering

Data Structures & Logic Tutor

Engineering

Dry-run traces, K-Maps, truth tables, and reference implementations

Autonomous Code Agent (Self-Healing)

Engineering

Minimal safe bug patches, traceback analysis, and verification steps

Super Vision

Vision

OCR, chart inspection, and grounded visual analysis using llava

UI/UX Vision Critic

Vision

Audits interface screenshots for hierarchy, contrast, and CSS fixes

AI Image & Thumbnail Creator

Creative

Generates 1280x720 (16:9) HD thumbnails from natural scene descriptions

YouTube Vlog Assistant

Creative

Hooks, shot-by-shot outlines, SEO titles, descriptions, and tags

Gen-Z Video Scripter

Creative

Short-form kinetic text scripts with 2-second hooks and beat timelines

Community & Visual Aid Creator

Education

Bilingual (Hindi + English) charts, posters, and community timetables

Auto-Recruitment Document Formatter

Career

ATS-friendly resume refinement and local image/PDF compression scripts

Weekend Chill Mode

Lifestyle

Curated movie, series, gaming, and weekend activity recommendations

PC Commander (Local Automation)

Utility

Launches local desktop workflows (VS Code, Notepad, Calculator, study tabs)

🧪 Independent Local SLM Lab Benchmark (phi3:mini)

This project includes a built-in Run local lab report benchmark engine that executes foundational queries offline on the local SLM and exports a timestamped report.

Terminal Execution Commands

# 1. Verify Ollama installation
ollama --version

# 2. Download and run the local Small Language Model
ollama run phi3:mini


Benchmark Prompts Tested Locally

What is Python?

Explain Flask in simple words.

Give me 5 task ideas for a to-do app.

Break "Learn Python" into 5 small steps.

📝 5-Line Lab Explanation Summary

Installed Ollama and executed the phi3:mini Small Language Model locally on the laptop.

Prompted the local model with four foundational programming and task-planning questions directly from the local runtime.

Processed all queries 100% offline using local GPU/CPU hardware resources without relying on ChatGPT or any paid online AI API.

Verified low-latency local token streaming, structured technical explanations, and complete on-device data privacy.

Confirmed that open-source Small Language Models can independently handle study, coding, and productivity workflows offline.

⚙️ Installation & Quick Start

1. Clone the Repository

git clone https://github.com/bhishmak529-sys/Bhishmak-SLM.git
cd Bhishmak-SLM


2. Install Python Dependencies

pip install --upgrade gradio PyPDF2 requests pillow gtts youtube-transcript-api python-docx beautifulsoup4 duckduckgo-search


3. Pull Recommended Local Models in Ollama

ollama pull phi3:mini
ollama pull llama3:latest
ollama pull qwen2.5-coder:1.5b
ollama pull llava:latest


4. Run the Application

python app.py


The interface will automatically find an available local port (default http://127.0.0.1:7920) and open in your web browser.

📂 Project Structure

Bhishmak-SLM/
├── app.py               # Main application (UI + Multi-Model Router + Hybrid RAG + Utilities)
├── README.md            # Project documentation and Lab Task summary
├── data/                # Auto-created local storage (chat history, memory, document vault)
└── generated/           # Auto-created local media outputs (thumbnails, voice clips, lab reports)


👤 Author

Bhishmak

GitHub: @bhishmak529-sys
