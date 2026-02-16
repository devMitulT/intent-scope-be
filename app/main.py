from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse,AnalyzeRequest,AnalyzeResponse,ScoreBlock
from app.llm_chain import chain
from app.vague_prompt_chain import prompt_chain
from app.url_chain import url_chain

from app.utils import fetch_clean_text
import os
from fastapi.middleware.cors import CORSMiddleware
from app.taxonomy import INTENTS, PERSONAS

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="LLM Query Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_cheack():
    return "App is working"

@app.post("/api/classify-query", response_model=QueryResponse)
async def classify_query(payload: QueryRequest):
    try:

        query =  prompt_chain.invoke({"query": payload.query} )
        print(query,"query")
        result =  chain.invoke({"query": query})

        intents = [i for i in result["intents"] if i in INTENTS]
        if not intents:
            intents = ["Informational"]

        personas = [p for p in result["personas"] if p in PERSONAS]
        if not personas:
            personas = ["Founder"]

        buying_level = result.get("buying_intent_level", "Low")

        keyword_type = result.get("keyword_type","Informational")

        return QueryResponse(
            intents=intents,
            personas=personas,
            reason=result["reason"],
            confidence=float(result["confidence"]),
            buying_intent_level=buying_level,
            keyword_type=keyword_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_page(request: AnalyzeRequest):
    print("request",request)
    try:
        content = fetch_clean_text(str(request.website_url))

        result = url_chain.invoke({
            "content": content,
            "keyword": request.keywords,
        })

        return AnalyzeResponse(
            seo=ScoreBlock(**result["seo"]),
            geo=ScoreBlock(**result["geo"]),
            aeo=ScoreBlock(**result["aeo"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))  

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
