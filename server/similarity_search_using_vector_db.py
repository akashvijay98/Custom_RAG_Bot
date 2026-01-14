import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_embeddings"
AWS_REGION = "us-east-1"

retry_config = Config(
    region_name=AWS_REGION,
    retries={
        'max_attempts': 20,
        'mode': 'adaptive'
    }
)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=retry_config
)

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1",
    region_name=AWS_REGION
)

client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Porsche knowledge base for car specifications, features, and brochure details.
    Useful for looking up specific facts before answering a question.
    """
    try:
       
        docs = vector_store.similarity_search(query, k=2)
        
        if not docs:
            return "No relevant information found."
        
        cleaned_docs = [d.page_content[:2000] for d in docs]
        
        return "\n\n".join(cleaned_docs)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

tools = [search_knowledge_base]

llm = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name=AWS_REGION,
    model_kwargs={"temperature": 0.0}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Porsche assistant.

Instructions:
1. Search specifically for the user's question.
2. Keep your answer SHORT (under 100 words).
3. Do NOT provide long lists. Summarize key points.
4. If you receive an error from the tool, STOP and tell the user.
5. If you find the answer, output it immediately.
"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=15,
    max_execution_time=60,
    handle_parsing_errors=True
)

def agentic_response(query_text: str) -> str:
    """
    Run the agentic workflow with multi-step reasoning
    Returns: str - Final answer from the agent
    """
    try:
        result = agent_executor.invoke({"input": query_text})
        return result["output"]
    except Exception as e:
        return f"Error running agent: {str(e)}"

def semantic_similarity_search(query_text: str, k: int = 3):
    """
    Direct vector similarity search without agent reasoning
    """
    try:
        return vector_store.similarity_search(query_text, k=k)
    except Exception as e:
        print(f"Search error: {e}")
        return []