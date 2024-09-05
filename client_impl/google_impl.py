

import os
import time

from llm_client_base import *
from typing import List

# pip install google-generativeai
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# config from .env
# GOOGLE_API_KEY
# HTTP_PROXY
# HTTPS_PROXY


class Gemini_Client(LlmClientBase):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self):
        super().__init__()

        api_key = os.getenv('GOOGLE_API_KEY')
        assert api_key is not None

        genai.configure(api_key=api_key)

    def role_convert_to_openai(self, role):
        if role == 'user':
            return 'user'
        elif role == 'model':
            return 'assistant'
        else:
            return 'unknown'

    def role_convert_from_openai(self, role):
        if role == 'user':
            return 'user'
        elif role == 'assistant':
            return 'model'
        else:
            return 'unknown'

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        force_calc_token_num = client_param.get('force_calc_token_num', False)

        system_message_list = [m for m in history if m['role'] == 'system']

        system_instruction = system_message_list[0]['content'] if system_message_list else None
        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature)
        messages = [{
            'role': self.role_convert_from_openai(m['role']),
            'parts': [m['content']]
        } for m in history
            if m['role'] != 'system'
        ]

        start_time = time.time()

        response = model.generate_content_async(
            messages,
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            stream=True)

        role = None
        result_buffer = ''
        finish_reason = None
        first_token_time = None

        async for chunk in await response:
            # print(chunk)
            if chunk.candidates:
                finish_reason = chunk.candidates[0].finish_reason.name
                delta_info = chunk.candidates[0]
                if delta_info.content.parts:
                    result_buffer += delta_info.content.parts[0].text
                    if first_token_time is None:
                        first_token_time = time.time()

                    role = self.role_convert_to_openai(delta_info.content.role)
                    yield LlmResponseChunk(
                        role=role,
                        delta_content=delta_info.content.parts[0].text,
                        accumulated_content=result_buffer,
                    )

        completion_time = time.time()

        usage = None
        if force_calc_token_num:
            usage = {
                'prompt_tokens': model.count_tokens(messages).total_tokens,
                'completion_tokens': model.count_tokens(result_buffer).total_tokens,
            }

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else completion_time - start_time,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = Gemini_Client()
    model_name = "gemini-1.5-pro-latest"
    history = [
        # {"role": "system", "content": "You are an assistant for home cooks. "},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
