

import os
import time

from llm_client_base import *
from typing import List
from openai import AsyncOpenAI

# config from .env
# OPENAI_API_KEY
# HTTP_PROXY
# HTTPS_PROXY


class OpenAI_Client(LlmClientBase):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self,
                 api_base_url=None,
                 api_key=None,
                 ):
        super().__init__()
        self.client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )

    async def close(self):
        await self.client.close()
        await super().close()

    def _extract_args(self, model_name, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature', None)
        max_tokens = model_param.pop('max_tokens', None)
        tools = model_param.pop('tools', None)
        functions = model_param.pop('functions', None)
        json_mode = model_param.get('json_mode', False)

        req_args = dict(
            model=model_name,
            stream=True,
            stream_options={'include_usage': True},
        )
        if temperature:
            req_args['temperature'] = temperature
        if json_mode:
            req_args['response_format'] = {"type": "json_object"}
        if max_tokens:
            req_args['max_tokens'] = max_tokens
        if tools:
            req_args['tools'] = tools
        elif functions:
            tools = [
                {
                    'type': 'function',
                    'function': f,
                }
                for f in functions
            ]
            req_args['tools'] = tools

        return req_args, model_param, client_param

    async def chat_async(self, model_name, history, model_param, client_param):
        req_args, left_model_param, left_client_param = self._extract_args(model_name, model_param, client_param)
        req_args['messages'] = history
        if left_model_param:
            req_args['extra_body'] = left_model_param

        req_args['stream'] = False

        start_time = time.time()

        response = await self.client.chat.completions.create(**req_args)
        message = response.choices[0].message
        # print(response)
        completion_time = time.time()

        usage = response.usage.dict()
        if usage.get('completion_tokens_details') and 'reasoning_tokens' in usage['completion_tokens_details']:
            usage['completion_tokens_details']['response_tokens'] = usage['completion_tokens'] - usage['completion_tokens_details']['reasoning_tokens']

        return LlmResponseTotal(
            role=message.role,
            accumulated_content=message.content,
            finish_reason=response.choices[0].finish_reason,
            system_fingerprint=response.system_fingerprint,
            real_model=response.model,
            usage=usage,
            first_token_time=None,
            completion_time=completion_time - start_time,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        req_args, left_model_param, left_client_param = self._extract_args(model_name, model_param, client_param)
        req_args['messages'] = history
        if left_model_param:
            req_args['extra_body'] = left_model_param

        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None
        real_model = None
        function_call_info_list = []

        async with await self.client.chat.completions.create(**req_args) as response:
            async for chunk in response:
                # print(chunk)
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info:
                        if delta_info.role:
                            role = delta_info.role
                        if delta_info.content:
                            result_buffer += delta_info.content

                            if first_token_time is None:
                                first_token_time = time.time()

                            yield LlmResponseChunk(
                                role=role or 'assistant',
                                delta_content=delta_info.content,
                                accumulated_content=result_buffer,
                            )
                        elif delta_info.function_call:
                            if first_token_time is None:
                                first_token_time = time.time()

                            if not function_call_info_list:
                                function_call_info_list = [{}]
                            if delta_info.function_call.name:
                                function_call_info_list[0]['name'] = delta_info.function_call.name
                            elif delta_info.function_call.arguments:
                                function_call_info_list[0]['arguments'] = function_call_info_list[0].get('arguments', '') + delta_info.function_call.arguments
                        elif delta_info.tool_calls:
                            if first_token_time is None:
                                first_token_time = time.time()

                            for tool_call in delta_info.tool_calls:
                                tool_call_idx = tool_call.index
                                while len(function_call_info_list) <= tool_call_idx:
                                    function_call_info_list.append({})

                                if tool_call.id:
                                    function_call_info_list[tool_call_idx]['id'] = tool_call.id
                                if tool_call.function.name:
                                    function_call_info_list[tool_call_idx]['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    function_call_info_list[tool_call_idx]['arguments'] = function_call_info_list[tool_call_idx].get('arguments', '') + tool_call.function.arguments

                if chunk.usage:
                    usage = chunk.usage.dict()
                if chunk.model:
                    real_model = chunk.model

        completion_time = time.time()

        tool_call_args_result = []
        for function_call_args in function_call_info_list:
            tool_call_args_result.append(LlmToolCallInfo(
                tool_call_id=function_call_args.get('id', None),
                tool_name=function_call_args.get('name', None),
                tool_args_json=function_call_args.get('arguments', None),
            ))

        yield LlmResponseTotal(
            role=role or 'assistant',
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            tool_calls=tool_call_args_result if tool_call_args_result else None,
            system_fingerprint=system_fingerprint,
            real_model=real_model,
            usage=usage or {},
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = OpenAI_Client(api_key=os.getenv('OPENAI_API_KEY'))
    model_name = "gpt-4o-mini"

    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
        # 'json_mode': True,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
