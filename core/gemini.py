import os
from google import genai
from google.genai import types
from typing import List, Any, Optional

class GeminiTextBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"

    def __str__(self):
        return self.text

class GeminiToolUseBlock:
    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input
        self.type = "tool_use"

class GeminiMessage:
    def __init__(self, content: list, stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason

class Gemini:
    def __init__(self, model: str):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model

    def _convert_to_gemini_contents(self, messages: list) -> list[types.Content]:
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(types.Part.from_text(text=block["text"]))
                        elif block.get("type") == "tool_use":
                            parts.append(types.Part.from_function_call(
                                name=block["name"],
                                args=block["input"]
                            ))
                        elif block.get("type") == "tool_result":
                            parts.append(types.Part.from_function_response(
                                name=block.get("name", "unknown_tool"),
                                response={"result": block["content"]}
                            ))
                    else:
                        parts.append(types.Part.from_text(text=str(block)))
            
            gemini_contents.append(types.Content(role=role, parts=parts))
        return gemini_contents

    def add_user_message(self, messages: list, message):
        content = message
        if hasattr(message, "content"):
            content = message.content
        
        messages.append({
            "role": "user",
            "content": content,
        })

    def add_assistant_message(self, messages: list, message):
        content = message
        if hasattr(message, "content"):
            content = message.content
        
        messages.append({
            "role": "assistant",
            "content": content,
        })

    def text_from_message(self, message: Any):
        if hasattr(message, "content"):
            return "\n".join(
                [block.text for block in message.content if getattr(block, "type", None) == "text"]
            )
        return str(message)

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> GeminiMessage:
        contents = self._convert_to_gemini_contents(messages)
        
        gemini_tools = None
        if tools:
            declarations = []
            for tool in tools:
                declarations.append(types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["input_schema"]
                ))
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = types.GenerateContentConfig(
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=gemini_tools,
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config
        )

        content_blocks = []
        stop_reason = "end_turn"
        
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content_blocks.append(GeminiTextBlock(part.text))
                    if part.function_call:
                        content_blocks.append(GeminiToolUseBlock(
                            id=part.function_call.name,
                            name=part.function_call.name,
                            input=part.function_call.args
                        ))
                        stop_reason = "tool_use"
        
        return GeminiMessage(content_blocks, stop_reason)
