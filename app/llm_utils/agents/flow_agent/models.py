from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict


class NodeData(BaseModel):
    label: str
    summary: str
    thought_ids: List[str]
    
class EdgeData(BaseModel):
    label: str

class Node(BaseModel):
    id: str
    data: NodeData

class Edge(BaseModel):
    id: str
    source: str
    target: str
    data: EdgeData
    
class ResultNode(Node):
    position: dict[str, int]
    type: str
    
class ResultEdge(Edge):
    type: str
    
class SelfReflectionAgentResult(BaseModel):
    nodes: List[Node] = Field(..., description="The nodes of the graph, with name of the node as instructed")
    edges: List[Edge] = Field(..., description="The edges of the graph, with name of the edge as instructed")
    message: str = Field(..., description="Description ofthe task performed by the agent")
    
class SelfReflectionAgentConnectorResult(BaseModel):
    edges: List[Edge] = Field(..., description="The edges that interconnect the themes, emotions, and goals")
    
class SelfReflectionAgentFinalResult(BaseModel):
    nodes: List[ResultNode] = Field(..., description="The nodes of the graph")
    edges: List[ResultEdge] = Field(..., description="The edges of the graph")

class SelfReflectionAgentState(TypedDict):
    user_id: str
    messages: Annotated[List[BaseMessage], operator.add]
    thoughts: List[Dict[str, Any]]
    max_themes: int
    max_emotions: int
    max_goals: int
    themeNodeResult: SelfReflectionAgentResult
    emotionNodeResult: SelfReflectionAgentResult
    goalNodeResult: SelfReflectionAgentResult
    connectorNodeResult: SelfReflectionAgentConnectorResult
    finalResult: SelfReflectionAgentFinalResult