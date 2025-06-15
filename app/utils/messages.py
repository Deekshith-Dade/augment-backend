from typing import List, Dict
from langchain_core.messages import BaseMessage
import uuid

def process_messagse_aisdk(messages: List[BaseMessage]) -> List[Dict]:
    """
    Process the messages to be used in the AI SDK.
    """
    processed_messages = []
    tool_parts = {}
    for message in messages:
        if message.type == "ai":
            # tool call
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_parts[tool_call['id']] = tool_call
            else:
                ai_message = {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": message.content,
                        "parts": [{
                            "type": "text",
                            "text": message.content
                        }]
                    }
                if tool_parts:
                    parts = []
                    tool_call_index = 0
                    for tool_call_id, tool_call in tool_parts.items():
                        parts.append({
                            "type": "tool_invocation",
                            "toolInvocation": {
                                "state": "result",
                                "step": tool_call_index,
                                "toolCallId": tool_call_id,
                                "args": tool_call['args'],
                                "toolName": tool_call['name'],
                                "result": tool_call['result']
                            }
                        })
                        tool_call_index += 1
                    ai_message['parts'].extend(parts)
                processed_messages.append(ai_message)
                tool_parts = {}
                
        elif message.type == "tool":
            # content from tool calls
            tool_parts[message.tool_call_id]['result'] = message.content
            
        elif message.type == "human":
            # human message
            processed_messages.append({
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": message.content,
                "parts": [{
                    "type": "text",
                    "text": message.content
                }]
            })
        
        
    return processed_messages