from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from langchain_community.embeddings import HuggingFaceEmbeddings
from my_llm import llm

import numpy as np
from dotenv import load_dotenv

_ = load_dotenv()


class GradedRagTool(BaseModel):
    binary_score: str = Field(description="文档与问题的相关性: 'yes' or 'no'")


class HallucinationsTool(BaseModel):
    binary_score: str = Field(description="问题与回答的相关性: 'yes' or 'no'")


class AnswerQuestionTool(BaseModel):
    binary_score: str = Field(description="问题与回答的相关性: 'yes' or 'no'")


"""
实现自省RAG.

该类用于根据相关性评分和生成回答，利用向量数据库进行检索，
并通过预训练语言模型对检索结果进行评估和回答生成。
"""


class GradeAndGenerateRagTool(object):
    """
    该类用于根据相关性评分和生成回答，利用向量数据库进行检索，
    并通过预训练语言模型对检索结果进行评估和回答生成。
    """

    def __init__(self, chroma_collection):
        """
        初始化Chroma数据库、嵌入模型和语言模型，
        以及用于评分的相关配置。
        """
        self.chroma_collection = chroma_collection
        self.embeding = HuggingFaceEmbeddings(model_name="./mxbai-embed-large-v1", model_kwargs={"device": "cpu"})
        self.llm = llm
        self.struct_llm_grader = self.llm.with_structured_output(GradedRagTool)
        self.struct_llm_halluciation = self.llm.with_structured_output(HallucinationsTool)
        self.struct_llm_answer = self.llm.with_structured_output(AnswerQuestionTool)

    def embed_dim(self, text):
        """
        计算文本的嵌入表示。

        参数:
        text (str): 输入的文本。

        返回:
        list: 文本的嵌入表示作为列表。
        """
        return self.embeding.embed_query(text)
    def search_vector(self, question):
        """
        根据问题在Chroma数据库中搜索相似的文档。

        参数:
        question (str): 输入的问题。

        返回:
        list[dict]: 包含最相似文档的元数据和文本。
        """
        # 将问题转换为向量
        query_embedding = self.embed_dim(question)

        # 在Chroma中进行相似度搜索
        results = self.chroma_collection.similarity_search_by_vector(
            query_embedding,
            k=1  # 返回最相关的单个文档
        )

        # 返回最相似的文档
        return results

    def grade(self, question, text):
        """
        评估检索到的文档与问题的相关性。

        参数:
        question (str): 输入的问题。
        text (str): 检索到的文档文本。

        返回:
        str: 相关性评分，'yes' 表示相关，'no' 表示不相关。
        """
        system_grade_prompt = """
        你是一名评估检索到文档与用户的问题相关性的评分员,不需要一个严格的测试,目标是过滤掉错误的检索,如果文档包含与用户问题相关的关键字或者语义,请评为相关,否则评为不相关
        """

        grade_message = [SystemMessage(content=system_grade_prompt)]
        grade_message.append(HumanMessage(content=f"问题:{question}\n文档:{text}"))
        result = self.struct_llm_grader.invoke(grade_message)
        return result.binary_score

    def generate(self, question, text):
        """
        根据检索到的文档生成回答。

        参数:
        question (str): 输入的问题。
        text (str): 检索到的文档文本。

        返回:
        str: 生成的回答。
        """
        generate_human_prompt = f"""
        你是问答任务的助理,使用以下检索到的上下文来回答问题,如果你不知道,就说你不知道,最多使用三句话,保持答案简洁,\n问题:{question}\n上下文:{text},\n答案:
        """
        human_prompt = ChatPromptTemplate.from_template(generate_human_prompt)
        end_prompt = human_prompt.format_messages(question=question, text=text)
        result = self.llm(end_prompt)
        return result.content

    def hallucinations(self, documents, answer):
        """
        评估生成的回答是否基于检索到的文档。

        参数:
        documents (str): 检索到的文档文本。
        answer (str): 生成的回答。

        返回:
        str: 是否基于文档生成，'yes' 表示不是，'no' 表示是。
        """
        hallucinations_prompt = "你是一名评估LLM生成是否基于一组检索到的事实的评分员，如果是基于检索到的事实回答则返回no，否则返回yes"
        hallucinations_message = [SystemMessage(content=hallucinations_prompt)]
        hallucinations_message.append(HumanMessage(content=f"文档:{documents}\n回答: {answer}"))
        result = self.struct_llm_halluciation.invoke(hallucinations_message)
        return result.binary_score

    def answer_question(self, question, answer):
        """
        评估回答是否解决了问题。

        参数:
        question (str): 输入的问题。
        answer (str): 生成的回答。

        返回:
        str: 是否解决问题，'yes' 表示是，'no' 表示否。
        """
        answer_question_prompt = """
        你是一名评分员，评估回答是否解决了问题，如果解决了则返回yes，否则返回no
        """
        answer_question_message = [SystemMessage(content=answer_question_prompt)]
        answer_question_message.append(HumanMessage(content=f"问题:{question}\n回答:{answer}"))
        result = self.struct_llm_answer.invoke(answer_question_message)
        return result.binary_score

    def rewrite_question(self, question):
        """
        重写问题，使其更精确，便于后续检索。

        参数:
        question (str): 输入的问题。

        返回:
        str: 重写后的问题。
        """
        rewrite_question_prompt = """
        你是一个将输入问题转化为优化的更好版本的问题重写器,用户向量数据库检索,查看输入并尝试推理潜在的语义意图或者含义
        """
        rewirte_message = [SystemMessage(content=rewrite_question_prompt)]
        rewirte_message.append(HumanMessage(content=f"问题:{question}"))
        result = self.llm.invoke(rewirte_message)
        return result.content
