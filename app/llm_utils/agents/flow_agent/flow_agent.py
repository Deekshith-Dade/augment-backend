import os
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage,  SystemMessage
from langchain_core.runnables import RunnableConfig
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from typing import Optional
from app.llm_utils.agents.flow_agent.models import SelfReflectionAgentState, SelfReflectionAgentResult, SelfReflectionAgentConnectorResult, SelfReflectionAgentFinalResult
from app.llm_utils.agents.flow_agent.prompts import (THEME_EXTRACTOR_SYSTEM_PROMPT, EMOTION_EXTRACTOR_SYSTEM_PROMPT, 
                     GOAL_EXTRACTOR_SYSTEM_PROMPT, CONNECTOR_EXTRACTOR_SYSTEM_PROMPT)
from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.llm_utils.agents.flow_agent.utils import get_similar_thoughts, pretty_thoughts, to_pgvector
from app.llm_utils.agents.flow_agent.models import ResultNode, ResultEdge
from sqlalchemy.orm import Session



os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "augument"

POSTGRES_POOL_SIZE = os.environ["POSTGRES_POOL_SIZE"]
POSTGRES_URL = os.environ["THOUGHT_AGENT_DB_URL"]


class FlowAgent:
    def __init__(self, llm=None, max_themes=2, max_emotions=2, max_goals=2):
        self.llm = llm
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self.graph: Optional[CompiledStateGraph] = None
        
        self.max_themes = max_themes
        self.max_emotions = max_emotions
        self.max_goals = max_goals
    
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
    
    async def _fetchThoughtsNode(self, state: SelfReflectionAgentState, config: RunnableConfig) -> SelfReflectionAgentState:
        try:
            user_id = state["user_id"]
            query = state["messages"][-1].content
            print(f"Query: {query}")
            embedding = await embed_text_openai(query)
            session = config.get("configurable").get("session")
            print(f"Session: {session}")
            results = await get_similar_thoughts(session, to_pgvector(embedding), 
                                user_id=user_id, 
                                top_k=5)
            thoughts = pretty_thoughts(results)
            print(f"Thoughts: {thoughts}")
            return {"thoughts": thoughts}
        except Exception as e:
            print(f"Error fetching thoughts: {e}")
            raise e

        
    async def _themeExtractorNode(self, state: SelfReflectionAgentState) -> SelfReflectionAgentState:
        messages = state["messages"]
        thoughts = state["thoughts"]
        max_themes = state["max_themes"]
        nodes = state["themeNodeResult"].nodes if "themeNodeResult" in state else []
        edges = state["themeNodeResult"].edges if "themeNodeResult" in state else []
        
        final_message = messages[-1].content if len(messages) > 0 else ""
        prompt = THEME_EXTRACTOR_SYSTEM_PROMPT.format(thoughts=thoughts, message=final_message, max_themes=max_themes, nodes=nodes, edges=edges)
        
    async def _themeExtractorNode(self, state: SelfReflectionAgentState) -> SelfReflectionAgentState:
        messages = state["messages"]
        thoughts = state["thoughts"]
        max_themes = state["max_themes"]
        nodes = state["themeNodeResult"].nodes if "themeNodeResult" in state else []
        edges = state["themeNodeResult"].edges if "themeNodeResult" in state else []
        
        final_message = messages[-1].content if len(messages) > 0 else ""
        prompt = THEME_EXTRACTOR_SYSTEM_PROMPT.format(thoughts=thoughts, message=final_message, max_themes=max_themes, nodes=nodes, edges=edges)
        
        structured_llm = self.llm.with_structured_output(SelfReflectionAgentResult)
        response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=f"Do the best job to extract the theme and form associations among them. Generate at most {max_themes} themes.")])
        

        return {"themeNodeResult": response}

    async def _emotionExtractorNode(self, state: SelfReflectionAgentState) -> SelfReflectionAgentState:
        messages = state["messages"]
        thoughts = state["thoughts"]
        max_emotions = state["max_emotions"]
        nodes = state["emotionNodeResult"].nodes if "emotionNodeResult" in state else []
        edges = state["emotionNodeResult"].edges if "emotionNodeResult" in state else []    
        
        final_message = messages[-1].content if len(messages) > 0 else ""
        prompt = EMOTION_EXTRACTOR_SYSTEM_PROMPT.format(thoughts=thoughts, message=final_message, max_emotions=max_emotions, nodes=nodes, edges=edges)
        
        structured_llm = self.llm.with_structured_output(SelfReflectionAgentResult)
        response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=f"Do the best job to extract the theme and form associations among them. Generate at most {max_emotions} emotions.")])
        

        return {"emotionNodeResult": response}

    async def _goalExtractorNode(self, state: SelfReflectionAgentState) -> SelfReflectionAgentState:
        messages = state["messages"]
        thoughts = state["thoughts"]
        max_goals = state["max_goals"]
        nodes = state["goalNodeResult"].nodes if "goalNodeResult" in state else []
        edges = state["goalNodeResult"].edges if "goalNodeResult" in state else []    
        
        final_message = messages[-1].content if len(messages) > 0 else ""
        prompt = GOAL_EXTRACTOR_SYSTEM_PROMPT.format(thoughts=thoughts, message=final_message, max_goals=max_goals, nodes=nodes, edges=edges)
        
        structured_llm = self.llm.with_structured_output(SelfReflectionAgentResult)
        response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=f"Do the best job to extract the theme and form associations among them. Generate at most {max_goals} goals.")])
        

        return {"goalNodeResult": response}


    async def _connectorNode(self, state: SelfReflectionAgentState) -> SelfReflectionAgentState:
        
        themeNodeResult = state["themeNodeResult"]
        emotionNodeResult = state["emotionNodeResult"]
        goalNodeResult = state["goalNodeResult"]
        message = state["messages"][-1].content if len(state["messages"]) > 0 else ""
        
        prompt = CONNECTOR_EXTRACTOR_SYSTEM_PROMPT.format(theme_nodes=themeNodeResult.nodes, theme_edges=themeNodeResult.edges, 
                                                        emotion_nodes=emotionNodeResult.nodes, emotion_edges=emotionNodeResult.edges, 
                                                        goal_nodes=goalNodeResult.nodes, goal_edges=goalNodeResult.edges, message=message)
        
        structured_llm = self.llm.with_structured_output(SelfReflectionAgentConnectorResult)
        response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Do the best job to connect the themes, emotions, and goals in a meaningful way.")])
        
        idx = 0
        nodes = []
        for node in themeNodeResult.nodes:
            nodes.append(ResultNode(id=node.id, data=node.data, type="self_reflection_theme", position={"x": idx * 100, "y": 100}))
            idx += 1
        for node in emotionNodeResult.nodes:
            nodes.append(ResultNode(id=node.id, data=node.data, type="self_reflection_emotion", position={"x": idx * 100, "y": 100}))
            idx += 1
        for node in goalNodeResult.nodes:
            nodes.append(ResultNode(id=node.id, data=node.data, type="self_reflection_goal", position={"x": idx * 100, "y": 200}))
            idx += 1
            
        edges = []
        for edge in themeNodeResult.edges:
            edges.append(ResultEdge(id=f"theme-{edge.id}", source=edge.source, target=edge.target, data=edge.data, type="self_reflection_theme"))
        for edge in emotionNodeResult.edges:
            edges.append(ResultEdge(id=f"emotion-{edge.id}", source=edge.source, target=edge.target, data=edge.data, type="self_reflection_emotion"))
        for edge in goalNodeResult.edges:
            edges.append(ResultEdge(id=f"goal-{edge.id}", source=edge.source, target=edge.target, data=edge.data, type="self_reflection_goal"))
        for edge in response.edges:
            edges.append(ResultEdge(id=f"connector-{edge.id}", source=edge.source, target=edge.target, data=edge.data, type="self_reflection_connector"))
            
        finalResult = SelfReflectionAgentFinalResult(nodes=nodes, edges=edges)
        
        return {"connectorNodeResult": response, "finalResult": finalResult}    
    
    async def _build_graph(self) -> StateGraph:
        if self.graph is not None:
            return self.graph
        
        try:
            graph_builder = StateGraph(SelfReflectionAgentState)

            graph_builder.add_node("fetch_thoughts", self._fetchThoughtsNode)
            graph_builder.add_node("theme_extractor", self._themeExtractorNode)
            graph_builder.add_node("emotion_extractor", self._emotionExtractorNode)
            graph_builder.add_node("goal_extractor", self._goalExtractorNode)
            graph_builder.add_node("connector", self._connectorNode)
            graph_builder.add_edge(START, "fetch_thoughts")
            graph_builder.add_edge("fetch_thoughts", "theme_extractor")
            graph_builder.add_edge("fetch_thoughts", "emotion_extractor")
            graph_builder.add_edge("fetch_thoughts", "goal_extractor")
            graph_builder.add_edge("theme_extractor", "connector")
            graph_builder.add_edge("emotion_extractor", "connector")
            graph_builder.add_edge("goal_extractor", "connector")
            graph_builder.add_edge("connector", END)
            
            connection_pool = await self._get_connection_pool()
            if connection_pool:
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()
            else:
                checkpointer = None
                raise Exception("connection pool initialization failed")
            
            self.graph = graph_builder.compile(checkpointer=checkpointer, name=f"Flow Agent")
        except Exception as e:
            print(f"Error building graph: {e}")
            raise e
        
        return self.graph
    
    async def get_response(self, question: str, user_id: Optional[str] = None, session: Optional[Session] = None) -> str:
        
        if self.graph is None:
            self.graph = await self._build_graph()
        
        config = {"configurable": {"thread_id": f"{user_id}", "session": session}}
        state = {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "max_themes": self.max_themes,
            "max_emotions": self.max_emotions,
            "max_goals": self.max_goals
        }
        try:
            response = await self.graph.ainvoke(state, config=config)
            return response["finalResult"]
        except Exception as e:
            print(f"Error getting response: {e}")
            raise e
    