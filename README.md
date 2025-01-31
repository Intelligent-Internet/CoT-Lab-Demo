---
title: "CoT-Lab: Human-AI Co-Thinking Laboratory"
emoji: "🤖"
colorFrom: "blue"
colorTo: "gray"
sdk: "gradio"
python_version: "3.13"
sdk_version: "5.13.1"
app_file: "app.py"
models:
  - "deepseek-ai/DeepSeek-R1"
tags:
  - "writing-assistant"
  - "multilingual"
license: "mit"
---

# CoT-Lab: Human-AI Co-Thinking Laboratory
[Huggingface Spaces 🤗](https://huggingface.co/spaces/Intelligent-Internet/CoT-Lab) | [GitHub Repository 🌐](https://github.com/Intelligent-Internet/CoT-Lab-Demo)
[中文README](README_zh.md)

**Sync your thinking with AI reasoning models to achieve deeper cognitive alignment**
Follow, learn, and iterate the thought within one turn

## 🌟 Introduction
CoT-Lab is an experimental interface exploring new paradigms in human-AI collaboration. Based on **Cognitive Load Theory** and **Active Learning** principles, it creates a "**Thought Partner**" relationship by enabling:

- 🧠 **Cognitive Synchronization**  
  Slow-paced AI output aligned with human information processing speed
- ✍️ **Collaborative Thought Weaving**   
  Human active participation in AI's Chain of Thought


** This project is part of ongoing exploration. Under active development, discussion and feedback are welcome! **

## 🛠 Usage Guide
### Basic Operation
1. **Set Initial Prompt**  
   Describe your prompy in the input box (e.g., "Explain quantum computing basics")

2. **Adjust Cognitive Parameters**  
   - ⏱ **Thought Sync Throughput**: tokens/sec - 5:Read-aloud, 10:Follow-along, 50:Skim
   - 📏 **Human Thinking Cadence**: Auto-pause every X paragraphs (Default off - recommended for active learning)

3. **Interactive Workflow**  
   - Click `Generate` to start co-thinking, follow the thinking process
   - Edit AI's reasoning when it pauses - or pause it anytime with `Shift+Enter`
   - Use `Shift+Enter` to hand over to AI again

## 🧠 Design Philosophy
- **Cognitive Load Optimization**  
  Information chunking (Chunking) adapts to working memory limits, serialized information presentation reduces cognitive load from visual searching

- **Active Learning Enhancement**  
  Direct manipulation interface promotes deeper cognitive engagement

- **Distributed Cognition**  
  Explore hybrid human-AI problem-solving paradiam 

## 📥 Installation & Deployment
Local deployment is (currently) required if you want to work with locally hosted LLMs.  
Due to degraded performance of official DeepSeek API - We recommend seeking alternative API providers, or use locally hosted distilled-R1 for experiment.    

**Prerequisites**: Python 3.11+ | Valid [Deepseek API Key](https://platform.deepseek.com/) or OpenAI SDK compatible API.

### Clone repository

    git clone https://github.com/Intelligent-Internet/CoT-Lab-Demo
    cd CoT-Lab-Demo

### Add to your .env file

API_KEY, API_URL, API_MODEL

### Start virtual environment

    python3 -m venv env
    source env/bin/activate

For Windows

    python3 -m venv env
    .\env\Scripts\activate

### Install dependencies

    pip install -r requirements.txt

### Launch application

    python app.py



## 📄 License
MIT License © 2024 [ii.inc]

## Contact
yizhou@ii.inc (Dango233)
