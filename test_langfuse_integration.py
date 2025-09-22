#!/usr/bin/env python3
"""
Test script to verify Langfuse integration with Aegra/LangGraph
"""

import os
import asyncio
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify environment variables
print("Checking Langfuse configuration...")
print(f"LANGFUSE_LOGGING: {os.getenv('LANGFUSE_LOGGING')}")
print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
if public_key:
    print(f"LANGFUSE_PUBLIC_KEY: {public_key[:20]}...")
else:
    print("LANGFUSE_PUBLIC_KEY: Not set!")

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3001")
)

# Verify connection
if langfuse.auth_check():
    print("✅ Langfuse client authenticated successfully!")
else:
    print("❌ Failed to authenticate with Langfuse")
    exit(1)

# Create Langfuse callback handler
langfuse_handler = CallbackHandler()

# Define a simple LangGraph agent
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the graph
def create_test_graph():
    builder = StateGraph(State)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define agent node
    def agent(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Build graph
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    return builder.compile()

async def test_agent():
    """Test the agent with Langfuse tracing"""
    print("\n🚀 Testing LangGraph agent with Langfuse tracing...")

    # Create the graph
    graph = create_test_graph()

    # Test input
    test_input = {
        "messages": [HumanMessage(content="Hello! Can you tell me what 2+2 equals?")]
    }

    # Run with Langfuse tracing
    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "agent_name": "test_agent",
            "test_run": True
        }
    }

    # Execute the graph
    result = await graph.ainvoke(test_input, config=config)

    print(f"\n📝 Agent response: {result['messages'][-1].content}")

    # Flush Langfuse data
    langfuse.flush()
    print("\n✅ Trace data sent to Langfuse!")
    print(f"📊 Check your traces at: {os.getenv('LANGFUSE_HOST')}")

    return result

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_agent())

    print("\n🎉 Test completed! Check Langfuse UI for the trace.")
    print(f"   URL: {os.getenv('LANGFUSE_HOST')}/project/default")