# llm_client_wrapper

对于各家LLM API client的封装，方便横向对比。

## 设计思路

* 优先使用OpenAI的client，各家逐步支持OpenAI协议后会切换为通过OpenAI client调用。
* 尽量避免冗余封装和设计，这也导致在使用时遇到问题需要看下对应实现
* 在支持新feature时才进行设计优化，避免过早为未来的需求进行预留
* LLM API经常出现超时，为了减少超时情况，优先适配stream调用。
* 优先适配async模式。
* Api-key等使用环境变量输入。

## 目前已经支持的Feature

目前各家LLM的client更新仍较快，我尽量维持对于最新版本的适配。

* 标准chat调用（仅 stream + async方式）
* Token usage 字段的统一化
* System message
* 角色扮演模型的bot profile设定，及简单多轮对话模拟

## 尚未支持的Feature

支持的Feature与我已经对各家API进行横评的进度有关。

尚未支持的Feature欢迎大家贡献PR。

* Json Schema指定
* Tools调用，function calling
* Token count API
* 知识库相关API
* Context prefix cache
* VL模型，其他多模态数据的支持
* Assistant API，文件上传API

## 支持的LLM供应商列表

### 海外

* OpenAI
* Google
* Anthropic
* Mistral
* Reka

### 国内

* 智谱
* 阿里巴巴（阿里云灵积）
* 字节跳动（火山引擎）
* 百度
* 零一万物
* 深度求索
* 阶跃星辰
* Moonshot
* 百川智能
* Minimax 
* 腾讯
* 商汤
* 讯飞

