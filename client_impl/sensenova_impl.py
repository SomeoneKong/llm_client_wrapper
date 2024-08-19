

import os
import time

import llm_client_base

# pip install sensenova
import sensenova

# config from .env
# SENSENOVA_KEY_ID
# SENSENOVA_SECRET_KEY


class SenseNova_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True
    support_multi_bot_chat: bool = True

    def __init__(self):
        super().__init__()

        key_id = os.getenv('SENSENOVA_KEY_ID')
        secret_access_key = os.getenv('SENSENOVA_SECRET_KEY')
        assert key_id is not None

        sensenova.access_key_id = key_id
        sensenova.secret_access_key = secret_access_key

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']

        args = {}
        if 'max_tokens' in model_param:
            args['max_new_tokens'] = min(3072, model_param['max_tokens'])

        start_time = time.time()
        response = await sensenova.ChatCompletion.acreate(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            **args
        )

        result_buffer = ''
        usage = None
        role = None
        finish_reason = None
        first_token_time = None

        async for chunk_resp in response:
            if chunk_resp.status.code != 0 and chunk_resp.data.choices[0].finish_reason == 'sensitive':
                raise llm_client_base.SensitiveBlockError()
            assert chunk_resp.status.code == 0, f"error: {chunk_resp}"

            chunk = chunk_resp.data
            usage = chunk['usage']
            usage = {
                'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens'],
                'knowledge_tokens': usage['knowledge_tokens'],
            }
            choice0 = chunk['choices'][0]
            role = choice0['role']
            if choice0['finish_reason']:
                finish_reason = choice0['finish_reason']
            result_buffer += choice0['delta']
            if choice0['delta'] and first_token_time is None:
                first_token_time = time.time()

            yield {
                'role': role,
                'delta_content': choice0['delta'],
                'accumulated_content': result_buffer,
                'usage': usage,
            }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }

    async def _multi_bot_chat(
            self,
            model_name,
            bot_chat_history,
            group_chat_param,
            model_param,
            client_param,
        ):
        # https://platform.sensenova.cn/doc?path=%2Fchat%2FCharacterChat%2FChatCompletions.md
        model_list = ['SenseChat-Character-Pro', 'SenseChat-Character']
        assert model_name in model_list, f"model_name must be in {model_list}"

        bot_profile_dict = group_chat_param['bot_profile_dict']
        next_bot = group_chat_param['next_bot']

        assert len(bot_chat_history) > 0, "bot_chat_history must have at least one message"

        for message in bot_chat_history:
            assert message['bot_name'] in bot_profile_dict, f"bot_name {message['bot_name']} not in bot_profile_dict"

        character_settings = []
        for bot_name, bot_profile in bot_profile_dict.items():
            character_settings.append({
                'name': bot_name,
                'gender': '未知',
                'detail_setting': bot_profile,
            })

        first_other_bot = list(r for r in bot_profile_dict if r != next_bot)[0]
        role_setting = {
            'user_name': first_other_bot,
            'primary_bot_name': next_bot,
        }

        accumulated_history = []
        for message in bot_chat_history:
            accumulated_history.append({
                'name': message['bot_name'],
                'content': message['content'],
            })

        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', 1024)

        args = {}
        if 'max_tokens' in model_param:
            args['max_new_tokens'] = min(1024, max_tokens)

        # character 模型不支持流式返回

        start_time = time.time()
        response = await sensenova.CharacterChatCompletion.acreate(
            model=model_name,
            messages=accumulated_history,
            temperature=temperature,
            character_settings=character_settings,
            role_setting=role_setting,
            **args
        )

        result_buffer = ''
        finish_reason = None
        first_token_time = None

        if response.data.choices[0].finish_reason == 'sensitive':
            raise llm_client_base.SensitiveBlockError()

        chunk = response.data
        usage = chunk['usage']
        usage = {
            'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'knowledge_tokens': usage['knowledge_tokens'],
        }
        choice0 = chunk['choices'][0]
        if choice0['finish_reason']:
            finish_reason = choice0['finish_reason']
        result_buffer += choice0['message']
        if choice0['message'] and first_token_time is None:
            first_token_time = time.time()

        completion_time = time.time()

        yield {
            'bot_name': next_bot,
            'delta_content': choice0['message'],
            'accumulated_content': result_buffer,
        }

        yield {
            'bot_name': next_bot,
            'content': result_buffer,
            'call_detail': {
                'finish_reason': finish_reason,
                'usage': usage,
                'first_token_time': first_token_time - start_time if first_token_time else None,
                'completion_time': completion_time - start_time,
            }
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = SenseNova_Client()
    model_name = "SenseChat-Turbo"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
