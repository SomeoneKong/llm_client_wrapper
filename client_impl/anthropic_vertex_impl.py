import time
import os

from llm_client_base import *
from typing import List

# pip install anthropic[vertex]
from anthropic.lib.vertex import AsyncAnthropicVertex

# config from .env
# GOOGLE_APPLICATION_CREDENTIALS


class AnthropicVertex_Client(LlmClientBase):
    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'west'

    def __init__(self,
                 google_auth_json_file=None,
                 google_region='us-east5',
                 ):
        super().__init__()
        # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude?hl=zh-cn

        # 使用json文件凭据
        # https://console.cloud.google.com/apis/credentials
        # 需要权限 Consumer Procurement Entitlement Manager, Vertex AI User
        self.google_auth_json_file = google_auth_json_file or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        assert self.google_auth_json_file, 'Please provide google_auth_json_file'

        # token 有效期 1小时
        self.project_id, self.access_token = self._get_access_token()
        self.region = google_region

        self.client = AsyncAnthropicVertex(
            region=google_region,
            project_id=self.project_id,
            access_token=self.access_token,
        )

    def _get_access_token(self):
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(self.google_auth_json_file, scopes=[
            'https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        return credentials.project_id, credentials.token

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        max_tokens = model_param.pop('max_tokens', 1024 * 3)  # 必选项

        system_message_list = [m for m in history if m['role'] == 'system']
        system_prompt = system_message_list[-1]['content'] if system_message_list else []

        message_list = [m for m in history if m['role'] != 'system']

        current_message = None
        start_time = time.time()
        first_token_time = None
        async with self.client.messages.stream(
                model=model_name,
                messages=message_list,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
        ) as stream:
            async for delta in stream.__stream_text__():
                current_message = stream.current_message_snapshot
                if delta and first_token_time is None:
                    first_token_time = time.time()
                yield LlmResponseChunk(
                    role=current_message.role,
                    delta_content=delta,
                    accumulated_content=current_message.content[0].text,
                )

        completion_time = time.time()

        usage = {
            'prompt_tokens': current_message.usage.input_tokens,
            'completion_tokens': current_message.usage.output_tokens,
        }
        yield LlmResponseTotal(
            role=current_message.role,
            accumulated_content=current_message.content[0].text,
            finish_reason=current_message.stop_reason,
            real_model=current_message.model,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )

    async def close(self):
        await self.client.close()

    def convert_multimodal_message(self, content: list):
        # https://docs.anthropic.com/en/docs/build-with-claude/vision
        # https://docs.anthropic.com/en/api/messages-examples#vision

        assert MultimodalMessageUtils.check_content_valid(content), f"Invalid content {content}"

        content_part_list = []
        for content_part in content:
            if isinstance(content_part, MultimodalMessageContentPart_Text):
                content_part_list.append({
                    'type': 'text',
                    'text': content_part.text,
                })
            elif isinstance(content_part, MultimodalMessageContentPart_ImageUrl):
                assert False, "Not support ImageUrl"
            elif isinstance(content_part, MultimodalMessageContentPart_ImagePath):
                image = ImageFile.load_from_path(content_part.image_path)
                assert image.image_format in ['jpeg', 'png', 'gif', 'webp'], f"Invalid image format {image.image_format}"

                content_part_list.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': image.mime_type,
                        'data': image.image_base64,
                    }
                })

        return content_part_list

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        max_tokens = model_param.pop('max_tokens', 1024 * 3)  # 必选项

        system_message_list = [m for m in history if m['role'] == 'system']
        system_prompt = system_message_list[-1]['content'] if system_message_list else []

        message_list = []
        for message in history:
            if message['role'] == 'system':
                continue

            if isinstance(message['content'], str):
                message_list.append({
                    'role': message['role'],
                    'content': message['content'],
                })
            elif isinstance(message['content'], list):
                message_list.append({
                    'role': message['role'],
                    'content': self.convert_multimodal_message(message['content']),
                })

        current_message = None
        start_time = time.time()
        first_token_time = None
        async with self.client.messages.stream(
                model=model_name,
                messages=message_list,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
        ) as stream:
            async for delta in stream.__stream_text__():
                current_message = stream.current_message_snapshot
                if delta and first_token_time is None:
                    first_token_time = time.time()
                yield LlmResponseChunk(
                    role=current_message.role,
                    delta_content=delta,
                    accumulated_content=current_message.content[0].text,
                )

        completion_time = time.time()

        usage = {
            'prompt_tokens': current_message.usage.input_tokens,
            'completion_tokens': current_message.usage.output_tokens,
        }
        yield LlmResponseTotal(
            role=current_message.role,
            accumulated_content=current_message.content[0].text,
            finish_reason=current_message.stop_reason,
            real_model=current_message.model,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = AnthropicVertex_Client()
    # model_name = "claude-3-5-sonnet@20240620"
    model_name = "claude-3-5-sonnet"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
