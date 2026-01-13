import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_embeddings"
AWS_REGION = "us-east-2"

# Initialize Embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=AWS_REGION
)

# Connect to Vector DB
client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# Define Tools
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Porsche knowledge base for car specifications, features, and brochure details.
    Useful for looking up specific facts before answering a question.
    """
    try:
        docs = vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

tools = [search_knowledge_base]

# Initialize LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name=AWS_REGION,
    model_kwargs={"temperature": 0.0}
)

# Create Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Porsche assistant with access to detailed brochure information. 

When answering questions:
1. Use the search_knowledge_base tool to find relevant information
2. For complex questions (comparisons, multi-part queries), break them down and search multiple times
3. Synthesize information from multiple searches when needed
4. Be specific and cite details from the brochures
5. If information isn't found, say so clearly"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent =  create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=5,
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
    Returns: List[Document] - Raw documents from vector store
    """
    try:
        return vector_store.similarity_search(query_text, k=k)
    except Exception as e:
        print(f"Search error: {e}")
        return []