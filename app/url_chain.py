from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response_schemas = [
    ResponseSchema(
        name="seo",
        description="SEO analysis block with score (int), good (list), needs_improvement (list), improvement_needed (string)"
    ),
    ResponseSchema(
        name="geo",
        description="GEO analysis block with score (int), good (list), needs_improvement (list), improvement_needed (string)"
    ),
    ResponseSchema(
        name="aeo",
        description="AEO analysis block with score (int), good (list), needs_improvement (list), improvement_needed (string)"
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
prompt = PromptTemplate(
    template="""
You are an SEO, GEO, and AEO evaluator.

INPUT FORMAT:
- keyword: a SINGLE complete phrase (string)
- content: webpage text

KEYWORD NORMALIZATION RULE:
If the keyword appears as a list, array, or comma-separated characters,
reconstruct it into ONE continuous phrase before analysis.

Examples:
["d","e","v","e"] → "deve"
"d, e, v, e" → "deve"

You MUST:
- Treat the keyword as ONE phrase
- NEVER analyze individual letters
- NEVER output characters as separate keywords

Normalized Keyword:
"{keyword}"

ANALYSIS RULES:
- Evaluate ONLY for this full keyword phrase or its close semantic variants
- If the full keyword is missing → score must be low
- Do not assume metadata, schema, backlinks, or technical SEO
- Do not give generic advice

SCORING SCALE:
1–3 → Not optimized
4–5 → Weak
6–7 → Moderate
8–9 → Strong
10 → Fully optimized

SEO CHECK:
- Keyword in title, H1, H2, first 100 words
- Natural usage
- Topical depth for the keyword
- Visible internal links relevant to the keyword
- Intent alignment

GEO CHECK:
- Clear semantic explanation of the keyword
- Dedicated section about the keyword
- Structured headings and lists
- Chunkable paragraphs
- High contextual clarity

AEO CHECK:
- Direct answer for the keyword
- Question-based subheadings
- 40–60 word definition block
- Bullet or step explanations
- FAQ-style content
- Voice-search friendly phrasing
- Featured snippet suitability

OUTPUT RULES:
- Analyze ONLY for: "{keyword}"
- Do NOT mention individual letters
- Do NOT rewrite the keyword as characters
- Return ONLY valid JSON

Content:
{content}

{format_instructions}
""",
    input_variables=["content", "keyword"],
    partial_variables={"format_instructions": format_instructions},
)



url_chain = prompt | llm | parser
