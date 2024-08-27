# llm_client_wrapper

对于各家LLM API client的封装，方便横向对比。

## 设计思路

* 本项目并不定位于直接用于实际产品项目，只是测试工具与调用逻辑展示。
  * 为了方便移植部分代码，尽量避免做非必要的封装。
  * 并未处理所有长尾情况，在使用时遇到问题时请直接查看对应实现。
  * 在支持新feature时才进行设计优化，避免过早为未来的需求进行预留
* 优先使用OpenAI的client，各家逐步支持OpenAI协议后会切换为通过OpenAI client调用。
* 在长输入时LLM API经常超过60s，为了减少全链路中各环节的超时问题，优先适配stream调用。
* 优先适配async模式。
* Api-key等使用环境变量输入。
* 有些厂家的Python SDK实现较差，没有太多使用价值，此类如果可行则会直接通过HTTP进行请求。

## 目前已经支持的Feature

目前各家LLM的client更新仍较快，我尽量维持对于最新版本的适配。

* 标准chat调用（仅 stream + async方式）
* Token usage 字段的统一化
* System message
* 角色扮演模型的bot profile设定，及简单多轮对话模拟

支持的Feature与我已经对各家API进行横评的进度有关。

## 尚未支持的Feature

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

### API代理商及第三方推理平台

* OpenRouter
* Together
* SiliconFlow
