# Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import asyncio
import json
from typing import TypedDict, Annotated, List, Dict, Any, operator, Optional, AsyncGenerator
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from asgiref.sync import sync_to_async
from langgraph.types import StateSnapshot

from app.llm_utils.agents.thought_agent.tools import fetch_relevant_thoughts, get_thought_details
from app.utils.messages import process_messagse_aisdk

from dotenv import load_dotenv
import os
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown

console = Console()


from app.database.database import SessionLocal
from fastapi import Depends

os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "augument"

POSTGRES_POOL_SIZE = os.environ["POSTGRES_POOL_SIZE"]
POSTGRES_URL = os.environ["THOUGHT_AGENT_DB_URL"]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class ReactAgent:
    def __init__(self, llm=None, tools=None):
        self.llm = llm
        self.tools = tools
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.tool_node = ToolNode(self.tools)
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self.graph: Optional[CompiledStateGraph] = None
    
    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """ Get a PostgreSQL connection pool"""
        
        if self._connection_pool is None:
            try:
                max_size = int(POSTGRES_POOL_SIZE)
                
                self._connection_pool = AsyncConnectionPool(
                    POSTGRES_URL,
                    max_size=max_size,
                    open=False,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None
                    }
                )
                await self._connection_pool.open()
            except Exception as e:
                raise e
        
        return self._connection_pool
    
    async def _build_graph(self) -> StateGraph:
        if self.graph is not None:
            return self.graph
            
        try:
            workflow = StateGraph(AgentState)
        
            workflow.add_node("agent", self._call_model)
            workflow.add_node("tools", self.tool_node)
            
            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "agent")
            workflow.add_edge("agent", END)
            
            connection_pool = await self._get_connection_pool()
            if connection_pool:
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()
            else:
                checkpointer = None
                raise Exception("connection pool initialization failed")
            
            self.graph = workflow.compile(checkpointer=checkpointer, name=f"Thoughts Agent")
        except Exception as e:
            print(f"Error building graph: {e}")
            raise e
        
        return self.graph
    
    def _prepare_messages(self, system_prompt: str, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Prepare the messages for the model"""
        
        trimmed_messages = trim_messages(
            messages,
            strategy="last",
            token_counter=self.llm,
            max_tokens=3000,
            start_on="human",
            include_system=False,
        )
        
        return [{"role": "system", "content": system_prompt}] + trimmed_messages
            
    async def _call_model(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        
        system_message = """You are a helpful assistant that can use tools to answer questions.
When you need to use a tool, simply call it with the appropriate function call. 
Think step by step and use tools when necessary to provide accurate answers.

If you don't need any tools, provide a direct answer.
You can use these tools to answer the question.


ALWAYS, If you are using a thought, you should cite it in the format [Number](thought://thought_id), I can use this info to make user experience better.
For example if the thought's title is "The future of AI" and the id is "23523523fadgadgasedga" and another thought's id is "23523523fadgadgasedga", you should cite it as [1](thought://23523523fadgadgasedga) and [2](thought://23523523fadgadgasedga).
Always start your answer with "Bello"
"""

        full_messages = self._prepare_messages(system_message, messages)
        try:
            response = await self.llm_with_tools.ainvoke(full_messages)
        except Exception as e:
            print(e)
            return {"messages": [AIMessage(content="I'm sorry, I'm having trouble answering your question. Please try again.")]}
        
        return {"messages": [response]}
        
    def _should_continue(self, state: AgentState) -> str:
        """A condition edge to determine wether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    async def get_response(self, question: str, session_id: str, user_id: Optional[str] = None, db: Optional[Session] = None) -> str:
        
        if self.graph is None:
            self.graph = await self._build_graph()
            
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }
        
        config = {"configurable": {
            "thread_id":  f"{user_id}_{session_id}",
            "user_id": user_id,
            "session": db
            }}
        
        
        result = await self.graph.ainvoke(initial_state, config=config)
        
        messages = result["messages"]
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.content
        
        return "Couldn't find the answer you were looking for"
    
    async def get_stream_response(self, question: str, session_id: str, user_id: Optional[str] = None, db: Optional[Session] = None) -> AsyncGenerator[str, None]:
        
        if self.graph is None:
            self.graph = await self._build_graph()
            
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }
        
        config = {"configurable": {
            "thread_id": f"{user_id}_{session_id}",
            "user_id": user_id,
            "session": db
            }}
        
        draft_tool_calls = []
        tool_call_details = {}
        draft_tool_calls_index = -1
        
        try:
            async for chunk, metadata in self.graph.astream(input=initial_state, config=config, stream_mode="messages"):
                if chunk.type == "tool":
                    if chunk.tool_call_id:
                        full_tool_call = tool_call_details[chunk.tool_call_id]
                        full_tool_call['result'] = chunk.content
                        yield 'a:{{"toolCallId":"{id}", "toolName":"{name}", "args":{args}, "result":{result}}}\n'.format(
                            name=full_tool_call["name"], 
                            id=full_tool_call["id"], 
                            result=json.dumps(full_tool_call["result"]),
                            args=json.dumps(full_tool_call["arguments"])
                        )
                        
                elif chunk.type == "AIMessageChunk":
                    if chunk.response_metadata:
                        if chunk.response_metadata['finish_reason'] == "stop":
                            yield 'e:{{"finishReason":"{reason}","isContinued":false}}\n'.format(reason="stop")
                            continue # REMARK: keep a return here instead of continue fucked me up, GeneratorExit on the langsmith traces                        
                        if chunk.response_metadata['finish_reason'] == 'tool_calls':
                            for tool_call in draft_tool_calls:
                                tool_call_details[tool_call['id']] = tool_call
                                yield '9:{{"toolCallId":"{id}", "toolName":"{name}", "args":{args}}}\n'.format(
                                    id=tool_call["id"], 
                                    name=tool_call["name"], 
                                    args=tool_call["arguments"]
                                )
                    
                    if chunk.tool_call_chunks:
                        for tool_call in chunk.tool_call_chunks:
                            id = tool_call['id']
                            name = tool_call['name']
                            arguments = tool_call['args']
                            
                            if (id is None):
                                draft_tool_calls[draft_tool_calls_index]["arguments"] += arguments
                            else:
                                draft_tool_calls_index += 1
                                draft_tool_calls.append(
                                    {"id": id, "name": name, "arguments": ""}
                                )
                    else:
                        # await asyncio.sleep(1)
                        yield '0:{text}\n'.format(text=json.dumps(chunk.content))
                        
        except Exception as e:
            print(f"Error: {e}")
            yield 'd:Error: {e}\n'.format(e=e)
    
    async def get_session_history(self, session_id: str, user_id: str) -> List[BaseMessage]:
        if self.graph is None:
            self.graph = await self._build_graph()
        
        try:
            state: StateSnapshot = await sync_to_async(self.graph.get_state)(
                config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
            )
            
            messages = state.values['messages']
            processed_messages = process_messagse_aisdk(messages)
            return processed_messages
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    

async def main():
    db = SessionLocal()
    
    try:
        tools = [
            fetch_relevant_thoughts,
            get_thought_details
        ]
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        agent = ReactAgent(llm=llm, tools=tools)
        
        while True:
            question = input("Enter your question: ")
            print("-"*100)
            response = await agent.get_response(question=question, session_id="121", user_id="83172f77-5d45-4ec2-ac7e-13e3d0f26504", db=db)
            console.print(Markdown(response))
            print("-"*100)
    
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())