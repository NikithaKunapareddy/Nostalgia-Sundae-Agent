# Agentic RAG with MCP Servers - Complete Implementation for Google Colab
# Based on the article: https://becomingahacker.org/integrating-agentic-rag-with-mcp-servers-technical-implementation-guide-1aba8fd4e442
import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import time
import re
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
#import openai
from flask import Flask, request, jsonify
import threading
import requests
import google.generativeai as genai
from bot_prompt import BOT_PROMPTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================


# Set your Gemini API key directly in code (not recommended for production)
GEMINI_API_KEY = your_google_gemini_key

# MCP Server Configuration
MCP_SERVER_PORT = 5000  # Changed from 8080 to avoid conflicts
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "knowledge_base.index"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class MCPRequest:
    """MCP protocol request structure"""
    id: str
    method: str
    params: Dict[str, Any]

@dataclass
class MCPResponse:
    """MCP protocol response structure"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class AgentState:
    """Agent conversation state"""
    conversation_id: str
    user_id: str
    context: Dict[str, Any]
    memory: List[Dict]

# ============================================================================
# VECTOR DATABASE IMPLEMENTATION
# ============================================================================

class VectorDatabase:
    """Simple FAISS-based vector database for knowledge storage"""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents: Dict[str, Document] = {}
        self.index = None
        self.dimension = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.embedding_model.encode([text])[0]

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add document to the vector database"""
        embedding = self._get_embedding(content)

        if self.index is None:
            self.dimension = len(embedding)
            self.index = faiss.IndexFlatL2(self.dimension)

        # Store document
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding
        )
        self.documents[doc_id] = doc

        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        logger.info(f"Added document {doc_id} to vector database")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = self._get_embedding(query)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        doc_list = list(self.documents.values())

        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(doc_list) and idx >= 0:
                doc = doc_list[idx]
                results.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "similarity_score": float(1 / (1 + distance))  # Convert distance to similarity
                })

        return results

# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """Manages long-term and short-term memory for the agent"""

    def __init__(self):
        self.short_term_memory: Dict[str, List[Dict]] = {}
        self.long_term_memory = VectorDatabase()
        self.user_preferences: Dict[str, Dict] = {}


    def store_conversation(self, user_id: str, message: Dict[str, Any]):
        """Store conversation in short-term memory and extract user preferences/facts dynamically"""
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = []

        self.short_term_memory[user_id].append({
            **message,
            "timestamp": datetime.now().isoformat()
        })

        # Limit short-term memory size
        if len(self.short_term_memory[user_id]) > 5000:
            self.short_term_memory[user_id] = self.short_term_memory[user_id][-5000:]

        # --- Extract user preferences/facts from message ---
        if message.get("role") == "user":
            content = message.get("content", "")
            extracted = self.extract_user_facts(content)
            if extracted:
                self.update_user_preferences(user_id, extracted)

    def extract_user_facts(self, text: str) -> dict:
        """Dynamically extract user facts/preferences using Gemini LLM (not hardcoded regex)."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = (
                "Extract all facts, preferences, or personal information about the user from the following message. "
                "Infer both the keys and values directly from the message, without using any hardcoded or example keys. "
                "Return a JSON object where each key is a descriptive label for the fact or preference, and each value is the corresponding value(s) as stated or implied by the user. "
                "If nothing is found, return an empty JSON object.\n"
                f"User message: {text}"
            )
            response = model.generate_content(prompt)
            extracted = response.text.strip()
            # Clean JSON if needed
            if extracted.startswith('```'):
                extracted = extracted.split('```')[1]
                if extracted.startswith('json'):
                    extracted = extracted[4:]
            try:
                facts = json.loads(extracted)
                if isinstance(facts, dict):
                    return facts
            except Exception:
                pass
            return {}
        except Exception as e:
            logger.error(f"LLM-based user fact extraction failed: {e}")
            return {}

    def store_long_term_memory(self, user_id: str, qa_pair: str, topic: str = "general"):
        """Store Q&A in long-term vector memory"""
        memory_id = hashlib.md5(f"{user_id}_{qa_pair}_{time.time()}".encode()).hexdigest()
        self.long_term_memory.add_document(
            memory_id,
            qa_pair,
            {"user_id": user_id, "type": "Q&A", "topic": topic, "timestamp": datetime.now().isoformat()}
        )

    def retrieve_context(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant context from memory"""
        # Get short-term memory
        recent_context = self.short_term_memory.get(user_id, [])[-limit:]

        # Search long-term memory
        long_term_results = self.long_term_memory.search(query, top_k=3)

        return {
            "recent_context": recent_context,
            "long_term_context": long_term_results
        }


    def get_user_preferences(self, user_id: str) -> Dict:
        """Get stored user preferences (deduplicate all list values, dynamic keys)"""
        prefs = self.user_preferences.get(user_id, {})
        for k, v in prefs.items():
            if isinstance(v, list):
                prefs[k] = list(dict.fromkeys([str(d).strip().lower() for d in v if str(d).strip()]))
        return prefs

    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences, merging lists and avoiding overwrites for all keys."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        for k, v in preferences.items():
            if k in self.user_preferences[user_id]:
                # If both are lists, merge and deduplicate
                if isinstance(self.user_preferences[user_id][k], list) and isinstance(v, list):
                    merged = self.user_preferences[user_id][k] + v
                    # Deduplicate and preserve order
                    seen = set()
                    merged_unique = [x for x in merged if not (str(x).lower() in seen or seen.add(str(x).lower()))]
                    self.user_preferences[user_id][k] = merged_unique
                # If existing is list and new is single value, append if not present
                elif isinstance(self.user_preferences[user_id][k], list):
                    if str(v).lower() not in [str(x).lower() for x in self.user_preferences[user_id][k]]:
                        self.user_preferences[user_id][k].append(v)
                # If existing is value and new is list, make it a list and merge
                elif isinstance(v, list):
                    existing = self.user_preferences[user_id][k]
                    merged = [existing] + v
                    seen = set()
                    merged_unique = [x for x in merged if not (str(x).lower() in seen or seen.add(str(x).lower()))]
                    self.user_preferences[user_id][k] = merged_unique
                # If both are values, keep as list if different
                else:
                    if str(self.user_preferences[user_id][k]).lower() != str(v).lower():
                        self.user_preferences[user_id][k] = [self.user_preferences[user_id][k], v]
            else:
                self.user_preferences[user_id][k] = v

# ============================================================================
# MCP SERVER IMPLEMENTATION
# ============================================================================

class MCPServer:
    """MCP Server for knowledge base access"""

    def __init__(self, vector_db: VectorDatabase, memory_manager: MemoryManager):
        self.vector_db = vector_db
        self.memory_manager = memory_manager
        self.app = Flask(__name__)
        self.setup_routes()


    def setup_routes(self):
        """Setup Flask routes for MCP protocol and /ask endpoint"""

        @self.app.route('/mcp', methods=['POST'])
        def handle_mcp_request():
            try:
                data = request.json
                mcp_request = MCPRequest(
                    id=data.get('id', str(time.time())),
                    method=data.get('method'),
                    params=data.get('params', {})
                )

                response = self.process_request(mcp_request)
                return jsonify({
                    "id": response.id,
                    "result": response.result,
                    "error": response.error
                })
            except Exception as e:
                logger.error(f"Error processing MCP request: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy"})

        # Register /ask endpoint on the main app
        @self.app.route('/ask', methods=['POST'])
        def ask():
            data = request.get_json(force=True)
            user_message = data.get("message", "")
            try:
                # Use the global agent instance
                global agent
                response = agent.process_query(user_message)
                return jsonify({"response": response})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request and return response"""
        try:
            if request.method == "search":
                return self._handle_search(request)
            elif request.method == "memory_store":
                return self._handle_memory_store(request)
            elif request.method == "memory_search":
                return self._handle_memory_search(request)
            elif request.method == "get_preferences":
                return self._handle_get_preferences(request)
            elif request.method == "list_capabilities":
                return self._handle_list_capabilities(request)
            else:
                return MCPResponse(
                    id=request.id,
                    error=f"Unknown method: {request.method}"
                )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error=str(e)
            )

    def _handle_search(self, request: MCPRequest) -> MCPResponse:
        """Handle knowledge base search"""
        query = request.params.get("query", "")
        top_k = request.params.get("top_k", 5)

        results = self.vector_db.search(query, top_k)

        return MCPResponse(
            id=request.id,
            result={"results": results}
        )

    def _handle_memory_store(self, request: MCPRequest) -> MCPResponse:
        """Handle memory storage"""
        user_id = request.params.get("user_id")
        data = request.params.get("data")

        self.memory_manager.store_conversation(user_id, data)

        return MCPResponse(
            id=request.id,
            result={"status": "stored"}
        )

    def _handle_memory_search(self, request: MCPRequest) -> MCPResponse:
        """Handle memory search"""
        user_id = request.params.get("user_id")
        query = request.params.get("query")
        limit = request.params.get("limit", 10)

        context = self.memory_manager.retrieve_context(user_id, query, limit)

        return MCPResponse(
            id=request.id,
            result={"data": context}
        )

    def _handle_get_preferences(self, request: MCPRequest) -> MCPResponse:
        """Handle get user preferences"""
        user_id = request.params.get("user_id")
        preferences = self.memory_manager.get_user_preferences(user_id)

        return MCPResponse(
            id=request.id,
            result={"preferences": preferences}
        )

    def _handle_list_capabilities(self, request: MCPRequest) -> MCPResponse:
        """List server capabilities"""
        capabilities = {
            "methods": [
                {
                    "name": "search",
                    "description": "Search the knowledge base",
                    "params": {"query": "string", "top_k": "number"}
                },
                {
                    "name": "memory_store",
                    "description": "Store data in memory",
                    "params": {"user_id": "string", "data": "object"}
                },
                {
                    "name": "memory_search",
                    "description": "Search memory",
                    "params": {"user_id": "string", "query": "string", "limit": "number"}
                },
                {
                    "name": "get_preferences",
                    "description": "Get user preferences",
                    "params": {"user_id": "string"}
                }
            ]
        }

        return MCPResponse(
            id=request.id,
            result=capabilities
        )

    def start_server(self, port: int = MCP_SERVER_PORT):
        """Start the MCP server"""
        def run_server():
            self.app.run(host='0.0.0.0', port=port, debug=False)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info(f"MCP Server started on port {port}")
        return server_thread

# ============================================================================
# MCP CLIENT IMPLEMENTATION
# ============================================================================

class MCPClient:
    """MCP Client for communicating with MCP servers"""

    def __init__(self, server_url: str = f"http://localhost:{MCP_SERVER_PORT}"):
        self.server_url = server_url
        self.request_id = 0

    def _get_next_id(self) -> str:
        """Get next request ID"""
        self.request_id += 1
        return str(self.request_id)

    def request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send request to MCP server"""
        try:
            payload = {
                "id": self._get_next_id(),
                "method": method,
                "params": params or {}
            }

            response = requests.post(
                f"{self.server_url}/mcp",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            if result.get("error"):
                raise Exception(f"MCP Error: {result['error']}")

            return result.get("result", {})

        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            raise

# ============================================================================
# AGENTIC RAG IMPLEMENTATION
# ============================================================================

class AgenticRAG:
    """Main Agentic RAG system combining LLM with MCP tools"""

    def __init__(self, mcp_client: MCPClient, persona: str = None, persona_vars: dict = None):
        self.mcp_client = mcp_client
        self.state = None
        self.persona = persona  # e.g., 'delhi_mentor_male'
        self.persona_vars = persona_vars or {}

    def get_persona_prompt(self):
        if self.persona and self.persona in BOT_PROMPTS:
            prompt_template = BOT_PROMPTS[self.persona]
            try:
                return prompt_template.format(**self.persona_vars)
            except Exception:
                return prompt_template
        return None

    def _call_openai(self, messages, temperature: float = 0.1) -> str:
        """Call Gemini API with a dynamic, context-aware prompt (no hardcoded persona)."""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Use the messages as the prompt directly, for a neutral, context-driven response
            prompt = str(messages)
            response = model.generate_content(prompt)
            extracted = response.text.strip()
            # Clean JSON if needed
            if extracted.startswith('```'):
                extracted = extracted.split('```')[1]
                if extracted.startswith('json'):
                    extracted = extracted[4:]
            try:
                data = json.loads(extracted)
                response = data
            except Exception:
                response = extracted
                return extracted
            return response
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"Error: Unable to generate response. {str(e)}"

    def analyze_intent(self, user_query: str):
        """Analyze user intent and determine required actions"""
        system_prompt = """You are an AI assistant that analyzes user queries to determine what actions are needed.

        Available tools:
        - search: Search the knowledge base for relevant information
        - memory_search: Search user's conversation history and stored memories
        - get_preferences: Get user's stored preferences

        Analyze the user query and respond with a JSON object containing:
        {
            "needs_search": boolean,
            "needs_memory": boolean,
            "needs_preferences": boolean,
            "search_query": "optimized search query if needed",
            "complexity": "simple|complex",
            "reasoning": "explanation of analysis"
        }        """

        try:
            format = """
            {
            "needs_search": boolean,
            "needs_memory": boolean,
            "needs_preferences": boolean,
            "search_query": "optimized search query if needed",
            "complexity": "simple|complex",
            "reasoning": "explanation of analysis"
        }        """

            instruction = """You are an AI assistant that analyzes user queries to determine what actions are needed.
              Available tools:
              - search: Search the knowledge base for relevant information
              - memory_search: Search user's conversation history and stored memories
              - get_preferences: Get user's stored preferences"""
            user_message = f" Analyze this query: {user_query}, you have following tools: search, memory_search, get_preferences and respond in following json object " + format
            response = self._call_openai(str(user_message))
            print("response :::::: ", response)
            if isinstance(response, dict):
                return response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception as e:
                    logger.error(f"JSON parsing failed: {e}")
            # Fallback analysis
            return {
                "needs_search": True,
                "needs_memory": False,
                "needs_preferences": False,
                "search_query": user_query,
                "complexity": "simple",
                "reasoning": "Default analysis due to parsing error"
            }
        except Exception as e:
            logger.error(f"Intent analysis failed HERE: {e}")
            return {
                "needs_search": True,
                "needs_memory": False,
                "needs_preferences": False,
                "search_query": user_query,
                "complexity": "simple",
                "reasoning": "Fallback due to error"
            }

    def search_knowledge(self, query: str, top_k: int = 5) -> str:
        """Search knowledge base via MCP. If no relevant info, return empty string for open-domain LLM fallback."""
        try:
            response = self.mcp_client.request("search", {
                "query": query,
                "top_k": top_k
            })

            docs = response.get("results", [])
            if not docs:
                return ""  # No relevant info, trigger open-domain LLM fallback

            # Format results for LLM
            formatted_results = []
            for i, doc in enumerate(docs[:3], 1):  # Limit to top 3
                content = doc["content"][:500]  # Truncate for brevity
                score = doc.get("similarity_score", 0)
                formatted_results.append(f"Document {i} (relevance: {score:.2f}):\n{content}")

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}, using open-domain LLM fallback")
            return ""  # On error, also trigger open-domain LLM fallback

    def get_memory_context(self, user_id: str, query: str) -> str:
        """Get relevant memory context via MCP"""
        try:
            response = self.mcp_client.request("memory_search", {
                "user_id": user_id,
                "query": query,
                "limit": 50
            })

            context_data = response.get("data", {})
            recent = context_data.get("recent_context", [])
            long_term = context_data.get("long_term_context", [])

            if not recent and not long_term:
                return "No relevant conversation history found."

            formatted_context = []

            if recent:
                formatted_context.append("Recent conversation:")
                for msg in recent[-10:]:  # Last 3 messages
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:200]
                    formatted_context.append(f"  {role}: {content}")

            if long_term:
                formatted_context.append("\nRelevant past conversations:")
                for mem in long_term[:3]:  # Top 2 relevant memories
                    content = mem["content"][:200]
                    formatted_context.append(f"  {content}")

            return "\n".join(formatted_context)

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return "No conversation history available."

    def process_query(self, user_query: str, user_id: str = "default_user") -> str:
        """Main query processing pipeline (short, focused, preference-based responses)"""
        logger.info(f"Processing query: {user_query}")

        # Step 1: Analyze intent
        intent = self.analyze_intent(user_query)
        logger.info(f"Intent analysis: {intent}")

        # Step 2: Gather context based on intent
        context_parts = []

        if intent.get("needs_search", False):
            search_query = intent.get("search_query", user_query)
            knowledge_context = self.search_knowledge(search_query)
            context_parts.append(f"Knowledge Base Information:\n{knowledge_context}")

        if intent.get("needs_memory", False):
            memory_context = self.get_memory_context(user_id, user_query)
            context_parts.append(f"Conversation Context:\n{memory_context}")

        # Always attempt to inject all user facts/preferences if any exist and intent suggests, or if the query is about the user
        try:
            prefs = self.mcp_client.request("get_preferences", {"user_id": user_id}).get("preferences", {})
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            prefs = {}

        # --- Short, focused, dynamic preference-based response logic ---
        # If the user expresses a negative mood (anxious, depressed, sad, stressed, upset, etc.), suggest their own coping preference
        negative_moods = [
            "anxious", "depressed", "sad", "stressed", "upset", "down", "unhappy", "worried", "blue", "low", "overwhelmed", "frustrated", "angry", "lonely", "hopeless", "discouraged", "tired", "burned out"
        ]
        user_query_lc = user_query.lower()
        if any(mood in user_query_lc for mood in negative_moods) and prefs:
            found = None
            found_mood = None
            for k, v in prefs.items():
                v_str = str(v).lower()
                for mood in negative_moods:
                    if mood in v_str:
                        # Try to extract what the user likes to do/eat/experience when feeling that mood
                        import re
                        match = re.search(r"like[s]? (.+?) when i'?m " + re.escape(mood), v_str)
                        if match:
                            found = match.group(1).strip()
                            found_mood = mood
                            break
                if found:
                    break
            if found:
                response = f"Maybe {found}."
            else:
                # Fallback: suggest any preference/fact that is positive and not a dislike
                for k, v in prefs.items():
                    v_str = str(v).lower()
                    if v_str.startswith("i like "):
                        found = v_str[7:].strip()
                        response = f"Maybe {found}."
                        break
                else:
                    response = None
            if response:
                # Store conversation
                try:
                    self.mcp_client.request("memory_store", {
                        "user_id": user_id,
                        "data": {
                            "role": "user",
                            "content": user_query
                        }
                    })
                    self.mcp_client.request("memory_store", {
                        "user_id": user_id,
                        "data": {
                            "role": "assistant",
                            "content": response
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to store conversation: {e}")
                return response

        # Otherwise, keep responses short and focused
        used_kb = False
        if not context_parts:
            logger.info("No relevant context found, falling back to LLM response")
            used_kb = False
            # Only inject user preferences/context if the user query contains a negative mood
            negative_moods = [
                "anxious", "depressed", "sad", "stressed", "upset", "down", "unhappy", "worried", "blue", "low", "overwhelmed", "frustrated", "angry", "lonely", "hopeless", "discouraged", "tired", "burned out"
            ]
            user_query_lc = user_query.lower()
            pref_context = ""
            if any(mood in user_query_lc for mood in negative_moods) and prefs:
                pref_lines = []
                for k, v in prefs.items():
                    key_label = k.replace('_', ' ').capitalize()
                    if isinstance(v, list):
                        value_str = ", ".join(str(i) for i in v)
                    else:
                        value_str = str(v)
                    pref_lines.append(f"{key_label}: {value_str}")
                if pref_lines:
                    pref_context = "\nUser preferences/context: " + "; ".join(pref_lines)
            # Short, direct system prompt
            system_prompt = (
                "You are a proactive, friendly, and engaging AI assistant with a unique personality.\n"
    "Always reply in one short, conversational sentence that feels dynamic and personal.\n"
    "If the user has shared preferences or context, weave them naturally into your answer.\n"
    "Never give one-word or generic responses—be warm, proactive, and a little playful.\n"
    "Never say you don't have information or suggest searching elsewhere.\n"
            )
            user_prompt = user_query + pref_context
            persona_prompt = self.get_persona_prompt()
            if persona_prompt:
                response = self._call_openai(system_prompt + "\n" + persona_prompt + "\n" + user_prompt)
            else:
                response = self._call_openai(system_prompt + "\n" + user_prompt)
        else:
            used_kb = True
            context_str = "\n\n".join(context_parts) if context_parts else "No additional context available."
            # Custom system prompt: do NOT mention irrelevant technical KB context
            system_prompt = (
                "You are a helpful AI assistant with access to knowledge base and conversation history.\n\n"
                "Instructions:\n"
                "- Use the provided context to answer user questions accurately.\n"
                "- If the context doesn't contain relevant information, answer using your own knowledge, but do NOT mention the knowledge base or its contents.\n"
                "- Never say you don't have information or suggest searching elsewhere.\n"
                "- Provide short, direct, and friendly answers.\n"
                "- Be conversational.\n"
            )
            user_prompt = f"Context Information:\n  {context_str}\n\n  User Question: {user_query}\n\n  Please provide a short, helpful response based on the available context."
            response = self._call_openai(system_prompt + "\n" + user_prompt)
        # Step 4: Store conversation in memory
        try:
            self.mcp_client.request("memory_store", {
                "user_id": user_id,
                "data": {
                    "role": "user",
                    "content": user_query
                }
            })

            self.mcp_client.request("memory_store", {
                "user_id": user_id,
                "data": {
                    "role": "assistant",
                    "content": response
                }
            })
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")

        return response

# ============================================================================
# INITIALIZATION AND SETUP
# ============================================================================

def setup_knowledge_base(vector_db: VectorDatabase):
    """Setup initial knowledge base with sample documents"""

    # Sample documents about Agentic RAG and MCP
    documents = [
        {
            "id": "agentic_rag_overview",
            "content": """Agentic RAG is an advanced form of Retrieval-Augmented Generation that incorporates AI agents to orchestrate the retrieval and generation process. Unlike traditional RAG, which performs static one-shot retrieval, agentic RAG systems can plan multi-step queries, use various tools, and adapt their strategy based on query complexity and intermediate results. The agent can decide when and how to retrieve information, which sources to use, and can even verify or cross-check information before generating the final answer.""",
            "metadata": {"topic": "agentic_rag", "type": "overview"}
        },
        {
            "id": "mcp_protocol",
            "content": """Model Context Protocol (MCP) is an open standard that standardizes how applications provide context to Large Language Models. It acts like a 'USB-C port for AI applications,' creating a universal interface to plug in external data and services. MCP defines a common protocol for AI assistants (clients) to communicate with external MCP servers that provide data or actions. This avoids the need for custom integrations for every new data source.""",
            "metadata": {"topic": "mcp", "type": "protocol"}
        },
        {
            "id": "agentic_rag_benefits",
            "content": """Agentic RAG provides several key benefits over traditional RAG: 1) Flexibility - can pull data from multiple knowledge bases or APIs, 2) Adaptability - adapts to different query contexts and user needs, 3) Improved Accuracy - iteratively refines retrieval results for higher quality answers, 4) Multi-step Reasoning - can formulate better search queries or perform multiple retrievals if needed, 5) Validation - can check retrieved facts and filter out irrelevant information.""",
            "metadata": {"topic": "agentic_rag", "type": "benefits"}
        },
        {
            "id": "mcp_servers",
            "content": """MCP servers are lightweight programs that expose specific capabilities through the standardized MCP protocol. Examples include: Knowledge Base Server (wrapping a vector database), Web Search Server (for web queries), Memory Server (for conversation context), File System Server (for document access), and API Integration Servers (for external services). Each server follows the same interaction rules, making the system modular and scalable.""",
            "metadata": {"topic": "mcp", "type": "servers"}
        },
        {
            "id": "implementation_steps",
            "content": """Implementing Agentic RAG with MCP involves: 1) Prepare knowledge base and create vector indexes, 2) Set up MCP server for retrieval interfacing with the knowledge base, 3) Configure MCP client/host environment to connect to servers, 4) Integrate retrieval calls in the agent using function calling or manual orchestration, 5) Implement multi-step retrieval for complex queries, 6) Establish knowledge update and storage processes.""",
            "metadata": {"topic": "implementation", "type": "steps"}
        }
    ]

    for doc in documents:
        vector_db.add_document(
            doc["id"],
            doc["content"],
            doc["metadata"]
        )

    logger.info(f"Added {len(documents)} documents to knowledge base")


# --- GLOBALS for API access ---
vector_db = None
memory_manager = None
mcp_server = None
mcp_client = None
agent = None

def main():
    """Main function to run the Agentic RAG system"""

    global vector_db, memory_manager, mcp_server, mcp_client, agent

    print("🚀 Starting Agentic RAG with MCP Servers...")

    # Initialize components
    vector_db = VectorDatabase()
    memory_manager = MemoryManager()

    # Setup knowledge base
    setup_knowledge_base(vector_db)

    # Start MCP server
    mcp_server = MCPServer(vector_db, memory_manager)
    server_thread = mcp_server.start_server()

    # Wait for server to start
    import time
    time.sleep(2)

    # Initialize MCP client and agent
    mcp_client = MCPClient()
    agent = AgenticRAG(mcp_client)

    print("✅ System initialized successfully!")
    print("🌐 REST API available at http://localhost:5000/ask (POST { 'message': ... })")

    # Interactive loop
    try:
        while True:
            user_input = input("🤖 Ask me anything: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\n🔍 Processing your query...")

            try:
                response = agent.process_query(user_input)
                print(f"\n📝 Response:\n{response}\n")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

            print("-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

    # --- Keep the server alive after CLI exit ---
    print("[INFO] CLI exited. Keeping REST API server alive. Press Ctrl+C to stop.")
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")

    return agent, mcp_client, vector_db, memory_manager

# ============================================================================
# GOOGLE COLAB SPECIFIC FUNCTIONS
# ============================================================================

def run_in_colab():
    """Function specifically designed for Google Colab execution"""

    # Install required packages (uncomment when running in Colab)
    """
    !pip install sentence-transformers faiss-cpu openai flask requests numpy
    """

    print("🔧 Installing required packages...")
    import subprocess
    import sys

    packages = [
        'sentence-transformers',
        'faiss-cpu',
        'openai==0.28.1',  # Using older version for compatibility
        'flask',
        'requests',
        'numpy'
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except:
            print(f"⚠️ Could not install {package}, please install manually")



    return main()

def demo_queries():
    """Run some demo queries to show the system working"""

    agent, mcp_client, vector_db, memory_manager = main()

    demo_questions = [
        "What is Agentic RAG?",
        "How does MCP protocol work?",
        "What are the benefits of using Agentic RAG?",
        "Can you explain the implementation steps?",
        "What types of MCP servers exist?"
    ]

    print("🎯 Running Demo Queries...\n")

    for i, question in enumerate(demo_questions, 1):
        print(f"📝 Demo Query {i}: {question}")
        try:
            response = agent.process_query(question)
            print(f"🤖 Response: {response}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
        print("-" * 80 + "\n")

    return agent, mcp_client, vector_db, memory_manager

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running in Google Colab
    try:
        import google.colab
        print("🔬 Detected Google Colab environment")
        run_in_colab()
    except ImportError:
        print("💻 Running in local environment")
        main()

# ============================================================================
# USAGE INSTRUCTIONS FOR GOOGLE COLAB
# ============================================================================

"""
GOOGLE COLAB USAGE INSTRUCTIONS:

1. First, set your OpenAI API key:
   - Replace "your-openai-api-key-here" with your actual OpenAI API key
   - Get one from: https://platform.openai.com/api-keys

2. Run the entire cell to install dependencies and start the system

3. Use the interactive interface to ask questions about:
   - Agentic RAG concepts and implementation
   - Model Context Protocol (MCP) servers
   - Benefits and use cases
   - Technical implementation details

4. Example function calls in Colab:

   # Initialize the system
   agent, client, db, memory = run_in_colab()

   # Ask a question
   response = agent.process_query("What is Agentic RAG?")
   print(response)

   # Run demo queries
   demo_queries()

   # Add custom knowledge
   db.add_document("custom_doc", "Your custom content here", {"topic": "custom"})

5. The system includes:
   - Vector database for knowledge storage
   - MCP server for standardized communication
   - Memory management for conversation context
   - Agentic reasoning for intelligent query processing

6. Features:
   - Multi-step reasoning
   - Context-aware responses
   - Memory persistence across conversations
   - Modular architecture with MCP protocol
   - Real-time knowledge base updates
"""