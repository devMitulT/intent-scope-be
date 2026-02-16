from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()
response_schemas = [
   ResponseSchema(
    name="enhanced_prompt",
    description=(
        "A rewritten and enriched version of the user's original query that is optimized "
        "for LLM processing. It should be more specific, structured, and context-aware, "
        "including inferred business intent, target persona, and key entities, while "
        "preserving the original meaning. This enhanced prompt should be suitable for "
        "downstream tasks such as semantic search, RAG retrieval, agent workflows, or "
        "advanced LLM reasoning."
    )
)
]


parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions() 
prompt = ChatPromptTemplate.from_template("""
You are a B2B query rewriting assistant.

Your task is to transform a short or ambiguous user query into a clearer, more specific, and LLM-optimized version suitable for semantic search, RAG retrieval, or agent workflows.

Guidelines:
- Rewrite the query to be more explicit and structured
- Preserve the original meaning and intent
- Add inferred business context only when it is strongly implied
- Do not introduce new facts, tools, companies, or numbers that were not mentioned or implied
- Keep the enriched query to one or two sentences
- Use professional, neutral business language

{format_instructions}

Query: {query}
""").partial(format_instructions=format_instructions)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt_chain = prompt | llm | parser
