#!/bin/bash

# Start Aegra with Docker Compose
echo "🚀 Starting Aegra - Self-hosted LangGraph Platform Alternative"
echo "=================================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating .env from example..."
    cp .env.example .env 2>/dev/null || echo "⚠️  No .env.example found, using defaults"
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: No LLM API keys found in environment"
    echo "Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  Warning: TAVILY_API_KEY not found"
    echo "The research agent requires Tavily API key for web search"
fi

# Start services
echo "Starting PostgreSQL and Aegra..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check health
echo "Checking service health..."
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Aegra is running at http://localhost:8000"
    echo "📊 API docs available at http://localhost:8000/docs"
    echo ""
    echo "Available agents:"
    echo "  - research: Deep research agent with web search"
    echo "  - agent: Basic ReAct agent"
    echo ""
    echo "To view logs: docker compose logs -f aegra"
    echo "To stop: docker compose down"
else
    echo "⚠️  Aegra might still be starting up..."
    echo "Check logs with: docker compose logs aegra"
fi