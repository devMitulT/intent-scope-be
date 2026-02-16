from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from app.taxonomy import INTENTS, PERSONAS
from dotenv import load_dotenv
load_dotenv()
response_schemas = [
    ResponseSchema(
        name="intents",
        description=f"List of one or more from: {INTENTS}"
    ),
    ResponseSchema(
        name="personas",
        description=f"List of one or more from: {PERSONAS}"
    ),
    ResponseSchema(name="reason", description="Short business motivation"),
    ResponseSchema(name="confidence", description="0 to 1"),
    ResponseSchema(
        name="buying_intent_level",
        description="One of: Very Low, Low, Medium, High, Very High"
    ),
    ResponseSchema(
    name="keyword_type",
    description="Commercial or Informational"
)
]


parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions() 
prompt = ChatPromptTemplate.from_template("""
ou are a B2B query classifier.

Return:
- intents → list (one or more)
- personas → list (one or more)
- reason
- confidence
- keyword_type                           
- buying_intent_level
                                          

Rules:
- Use only allowed values
- Multiple intents/personas allowed if relevant
- Buying intent logic:
  Pricing/demo → Very High
  Best tools/compare → High
  Features → Medium
  General → Low
  Learning → Very Low

{format_instructions}

Query: {query}
""").partial(format_instructions=format_instructions)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

chain = prompt | llm | parser
