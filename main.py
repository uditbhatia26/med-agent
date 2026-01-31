import sys
import asyncio
import os
from typing import TypedDict, Annotated

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph_checkpoint_dynamodb.saver import DynamoDBSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# LLM Configuration
LLM_MODEL = 'gpt-4o-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'

# Document Processing Configuration
PDF_PATH = 'LLMs Questions.pdf'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_TOP_K = 4

# Database Configuration
DB_CONNECTION_STRING = os.getenv('SUPABASE_URL')
DEFAULT_THREAD_ID = '12'

# MCP Configuration
FASTMCP_TOKEN = os.getenv('FASTMCP_TOKEN')

llm = ChatOpenAI(model=LLM_MODEL)

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "https://weather-ca299b5b8522.fastmcp.app/mcp",
            "headers": {  
                "Authorization": f"Bearer {FASTMCP_TOKEN}",
            },  
        },
    }
)

# State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialization Functions
def initialize_llm() -> ChatOpenAI:
    """Initialize the language model"""
    return ChatOpenAI(model=LLM_MODEL)


def initialize_mcp_client() -> MultiServerMCPClient:
    """Initialize MCP client for external tools"""
    return MultiServerMCPClient({
        "weather": {
            "transport": "http",
            "url": "https://weather-ca299b5b8522.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {FASTMCP_TOKEN}"
            }
        },
    })


def initialize_rag_retriever():
    """Load PDF, create embeddings, and initialize retriever"""
    print(f"📄 Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(file_path=PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': RETRIEVER_TOP_K}
    )
    
    return retriever


retriever = initialize_rag_retriever()


@tool
async def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.
    
    Use this tool when the user asks factual or conceptual questions
    that might be answered from the stored documents about LLMs.
    
    Args:
        query: The search query to find relevant information
        
    Returns:
        String containing the query and relevant context chunks
    """
    result = await retriever.ainvoke(query)
    context = [doc.page_content for doc in result]
    
    # Return as formatted string for better compatibility
    return f"Context for query '{query}':\n\n" + "\n\n---\n\n".join(context)


async def load_all_tools(mcp_client: MultiServerMCPClient) -> list:
    """Load MCP tools and combine with local tools"""
    print("🔄 Loading MCP tools...")
    try:
        mcp_tools = await mcp_client.get_tools()
        print(f"✅ Loaded {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"⚠️  Failed to load MCP tools: {e}")
        mcp_tools = []
    
    all_tools = mcp_tools + [rag_tool]
    print(f"📦 Total tools available: {len(all_tools)}")
    
    return all_tools


def build_graph(llm_with_tools, tools) -> StateGraph:
    """Build the conversation graph with tools"""
    
    async def chat(state: ChatState):
        """Chat node that processes messages with tools"""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}
    
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("Chat", chat)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "Chat")
    graph.add_conditional_edges("Chat", tools_condition)
    graph.add_edge("tools", "Chat")
    
    return graph


async def run_chat_loop(chatbot):
    """Run the interactive chat loop"""
    print("\n" + "="*60)
    print("🤖 Chatbot Ready!")
    print("="*60)
    print("Commands: 'quit', 'q', or 'bye' to exit")
    print("="*60 + "\n")
    
    while True:
        user_query = input("\n👤 Human: ")

        if user_query.lower() in ['bye', 'q', 'quit']:
            print("\n👋 Goodbye!\n")
            break

        print("🤖 Assistant: ", end='', flush=True)
        
        try:
            response = chatbot.astream(
                {'messages': [HumanMessage(content=user_query)]},
                config={'configurable': {'thread_id': DEFAULT_THREAD_ID}},
                stream_mode='messages'
            )
            
            async for message_chunk, _ in response:
                if (isinstance(message_chunk, AIMessage) and message_chunk.content):
                    print(message_chunk.content, end='', flush=True)
            
            print()
            
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            print("💡 Tip: Try using a different thread_id or clearing message history")
            print()


async def main():
    """Main application entry point"""
    
    llm = initialize_llm()
    mcp_client = initialize_mcp_client()
    
    tools = await load_all_tools(mcp_client)

    llm_with_tools = llm.bind_tools(tools)

    graph = build_graph(llm_with_tools, tools)

    async with AsyncPostgresSaver.from_conn_string(DB_CONNECTION_STRING) as checkpoint:
        await checkpoint.setup()
        print("✅ Database ready")
        
        chatbot = graph.compile(checkpointer=checkpoint)
        
        await run_chat_loop(chatbot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
