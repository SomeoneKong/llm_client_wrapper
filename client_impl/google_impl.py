

import os
import time

from llm_client_base import *
from typing import List

# pip install google-generativeai
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PIL.Image
from google.generativeai import protos

# config from .env
# GOOGLE_API_KEY


class Gemini_Client(LlmClientBase):
    support_system_message: bool = True
    support_image_message: bool = True

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

    def get_safety_settings(self):
        return {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

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
            safety_settings=self.get_safety_settings(),
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
            prompt_token_num = model.count_tokens(messages).total_tokens
            completion_token_num = 0
            if result_buffer:
                completion_token_num = model.count_tokens(result_buffer).total_tokens
            usage = {
                'prompt_tokens': prompt_token_num,
                'completion_tokens': completion_token_num,
            }

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else completion_time - start_time,
            completion_time=completion_time - start_time,
        )


    def convert_multimodal_message(self, content: list, file_handle_list:list):
        assert MultimodalMessageUtils.check_content_valid(content), f"Invalid content {content}"

        content_part_list = []
        for content_part in content:
            if isinstance(content_part, MultimodalMessageContentPart_Text):
                content_part_list.append(content_part.text)
            elif isinstance(content_part, MultimodalMessageContentPart_ImageUrl):
                assert False, "Not support ImageUrl"
            elif isinstance(content_part, MultimodalMessageContentPart_ImagePath):
                image = ImageFile.load_from_path(content_part.image_path)
                assert image.mime_type in ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'], f"Unsupported image format {image.mime_type}"
                pil_image = PIL.Image.open(content_part.image_path)
                content_part_list.append(pil_image)
            elif isinstance(content_part, MultimodalMessageContentPart_AudioPath):
                # https://ai.google.dev/gemini-api/docs/audio?lang=python
                ext = os.path.splitext(content_part.audio_path)[1][1:]
                assert ext in ['mp3', 'wav', 'flac', 'ogg', 'aiff', 'aac'], f"Unsupported audio format {ext}"
                file = genai.upload_file(content_part.audio_path)
                content_part_list.append(file)
                file_handle_list.append(file)
            elif isinstance(content_part, MultimodalMessageContentPart_VideoPath):
                # https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video
                ext = os.path.splitext(content_part.video_path)[1][1:]
                assert ext in ['mp4', 'mpeg', 'mov', 'avi', 'mkv', 'webm', 'flv'], f"Unsupported video format {ext}"
                file = genai.upload_file(content_part.video_path)
                content_part_list.append(file)
                file_handle_list.append(file)

        return content_part_list

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        force_calc_token_num = client_param.get('force_calc_token_num', False)

        system_message_list = [m for m in history if m['role'] == 'system']

        system_instruction = system_message_list[0]['content'] if system_message_list else None
        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature)


        message_list = []
        file_handle_list = []
        for message in history:
            if message['role'] == 'system':
                continue

            if isinstance(message['content'], str):
                message_list.append({
                    'role': self.role_convert_from_openai(message['role']),
                    'parts': [message['content']]
                })
            elif isinstance(message['content'], list):
                message_list.append({
                    'role': self.role_convert_from_openai(message['role']),
                    'parts': self.convert_multimodal_message(message['content'], file_handle_list),
                })

        if len(file_handle_list) > 0:
            print(f'waiting for uploaded files to be processed ...')
            for file in file_handle_list:
                while True:
                    file_info = genai.get_file(file.name)

                    if file_info.state == protos.File.State.ACTIVE:
                        break

                    assert file_info.state != protos.File.State.FAILED, f'File {file.name} gemini process failed'

                    await asyncio.sleep(1)
            print(f'uploaded files processed')

        start_time = time.time()

        response = model.generate_content_async(
            message_list,
            generation_config=generation_config,
            safety_settings=self.get_safety_settings(),
            stream=True
        )

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
            prompt_token_num = model.count_tokens(message_list).total_tokens
            completion_token_num = 0
            if result_buffer:
                completion_token_num = model.count_tokens(result_buffer).total_tokens
            usage = {
                'prompt_tokens': prompt_token_num,
                'completion_tokens': completion_token_num,
            }

        for file in file_handle_list:
            try:
                genai.delete_file(file.name)
            except Exception as e:
                print(f'Error deleting file {file.name}: {e}')

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
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
