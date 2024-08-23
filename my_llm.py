from langchain_openai import ChatOpenAI


"""
base_url：指定API的基础URL地址。
api_key：提供访问所需的密钥。
model：选择使用的模型版本为qwen2-72b-instruct。
temperature：设置生成文本的随机性程度为0，值越高输出越随机。
"""
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-46918778e42746498939ce47a7a01ce9",
    model="qwen2-72b-instruct",
    temperature=0,
)


# print(llm.invoke("你是谁"))

