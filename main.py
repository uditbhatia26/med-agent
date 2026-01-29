import sys
import asyncio
import os
from typing import TypedDict, Annotated

# Fix for Windows event loop compatibility
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
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "https://weather-ca299b5b8522.fastmcp.app/mcp",
            "headers": {  
                "Authorization": f"Bearer fmcp_pK2yvWhNnJ1EJde-9FOla2He9HtnR16o9dtM5U_5nWE",
            },  
        },
        
        "expense": {
            'transport': 'http',
            'url': 'https://labour-magenta-salmon.fastmcp.app/mcp',
            "headers": {  
                "Authorization": "Bearer fmcp_pK2yvWhNnJ1EJde-9FOla2He9HtnR16o9dtM5U_5nWE",
            },
        }
    }
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

loader = PyPDFLoader(file_path='LLMs Questions.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)
# vector_store.save_local()
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})


@tool
async def rag_tool(query):

    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    result = await retriever.ainvoke(query)

    context = [doc.page_content for doc in result]

    return {
        'query': query,
        'context': context,
    }


async def main():
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [rag_tool]

    
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def chat(state: ChatState):
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)

        return {'messages': [response]}


    graph = StateGraph(ChatState)
    graph.add_node("Chat", chat)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, "Chat")
    graph.add_conditional_edges("Chat", tools_condition)
    graph.add_edge("tools", "Chat")
    graph.add_edge("Chat", END)
    
    async with AsyncPostgresSaver.from_conn_string(conn_string=os.getenv('SUPABASE_URL')) as checkpoint:
        await checkpoint.setup()
        chatbot = graph.compile(checkpointer=checkpoint)

# Runn this code only for testing
        while True:
            user_query = input("Human: ")
            if user_query.lower() in ['bye', 'q', 'quit']:
                break
            
            response = chatbot.astream(
                {
                    'messages': [HumanMessage(content=user_query)]
                },
                config={
                    'configurable':{
                        'thread_id': '10'
                    }
                },
                stream_mode='messages'
            )
            
            async for message_chunk, _ in response:
                if isinstance(message_chunk, AIMessage) and message_chunk.content and not message_chunk.tool_calls:
                    print(message_chunk.content, end='', flush=True)
            print()

if __name__ == "__main__":
    asyncio.run(main())
