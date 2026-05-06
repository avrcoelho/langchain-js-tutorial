# LangChain JS Course Examples

This repository contains practical examples and exercises for learning [LangChain.js](https://js.langchain.com/), specifically focused on integrating with Google Gemini models.

## Project Overview

The project demonstrates core LangChain concepts:
- **LLM Interaction:** Simple invocation of chat models.
- **Prompt Templates:** Constructing dynamic prompts using templates and message roles.
- **Output Parsers:** Transforming raw model output into structured formats like lists or JSON objects using Zod.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Node.js installed, then run:

```bash
npm install
```

### 2. Configure Environment Variables

The project uses `dotenv` to manage sensitive information. You need to provide a Google API Key to interact with Gemini models.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open the newly created `.env` file.
3. Replace `"google-api"` with your actual Google API Key:
   ```env
   GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"
   ```

## How to Run the Examples

You can execute each JavaScript file using Node.js.

### Basic LLM Call
To see a simple "Hello World" interaction:
```bash
node llm.js
```

### Prompt Templates
To see how prompts are structured and formatted:
```bash
node prompt-template.js
```

### Output Parsers
This file contains several demonstrations of parsing logic (String, Comma-Separated List, Structured, and Zod).
*Note: The functions in this file are defined but not all are called by default. You can uncomment the desired function call at the bottom of the file to test specific parsers.*
```bash
node output-parsers.js
```
