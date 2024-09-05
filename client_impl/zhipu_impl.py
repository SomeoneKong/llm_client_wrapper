
import os

from llm_client_base import *
from typing import List
from .openai_impl import OpenAI_Client

# config from .env
# ZHIPU_API_KEY

# https://open.bigmodel.cn/dev/api#language


class Zhipu_Client(OpenAI_Client):
    support_system_message: bool = True
    support_chat_with_bot_profile_simple: bool = True

    server_location = 'china'

    def __init__(self):
        api_key = os.getenv('ZHIPU_API_KEY')
        assert api_key is not None

        super().__init__(
            api_base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=api_key,
        )

    async def _chat_with_bot_profile_simple(self, model_name, history, bot_profile_dict, model_param, client_param):
        assert model_name in ["charglm-3"], f"Unsupported model_name: {model_name}"

        bot_setting = {
            'user_name': bot_profile_dict['user']['name'],
            'bot_name': bot_profile_dict['bot']['name'],
            'user_info': bot_profile_dict['user']['content'],
            'bot_info': bot_profile_dict['bot']['content'],
        }

        raw_model_param = model_param.copy()
        raw_model_param['meta'] = bot_setting

        async for chunk in self.chat_stream_async(model_name, history, raw_model_param, client_param):
            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Zhipu_Client()
    model_name = "glm-4-flash"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
