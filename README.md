# REPORTLY API

This repository contains the API for the REPORTLY application, which generates report content using AI.

## Deployment to Vercel

1. Push this repository to GitHub
2. Connect your GitHub repository to Vercel
3. Set the following Environment Variables in Vercel:
   - `GOOGLE_API_KEY`: Your Google Generative AI API key

## Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your API key:
   ```
   # Windows
   set GOOGLE_API_KEY=your_api_key_here
   
   # Unix/Mac
   export GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the server:
   ```
   cd server
   python langchain_server.py
   ```
   
The server will be available at http://localhost:8000

## API Endpoints

- `GET /`: Health check endpoint
- `GET /test`: Test if the API is running and API key is configured
- `POST /generate-details`: Generate report content based on input
- `POST /echo`: Echo back the request for debugging
- `POST /manual-generate`: Manual parsing endpoint for troubleshooting 