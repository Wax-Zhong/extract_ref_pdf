from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
import os
import fnmatch
import pandas as pd
from utils.self_rag_tool import GradeAndGenerateRagTool

# 初始化OpenAI API密钥
os.environ['OPENAI_API_KEY'] = 'openai-api-key'


def load_pdf(file_path):
    """
    加载PDF文档。

    :param file_path: PDF文件的路径
    :return: 加载的文档列表
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def create_vectorstore(documents):
    """
    根据加载的文档创建Milvus向量数据库。

    :param documents: 加载的文档列表
    :return: Milvus向量数据库
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def extract_information(query, vectorstore):
    """
    从向量数据库中提取信息。

    :param query: 查询文本
    :param vectorstore: 向量数据库
    :return: 提取的信息
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vectorstore.as_retriever()
    )
    result = qa_chain.run(query)
    return result


def find_pdf_files(root_dir):
    """
    查找指定目录及其子目录中的所有PDF文件，并返回它们的绝对路径列表。

    :param root_dir: 包含PDF文件的根目录路径
    :return: PDF文件的绝对路径列表
    """
    pdf_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*.pdf'):
            pdf_files.append(os.path.join(dirpath, filename))
    return pdf_files


# 该方法用于验证PDF文件数量是否不为0且在100个以内，并抛出异常,从而避免OOM。
def validate_pdf_file(file_path):
    if not file_path:
        raise ValueError("No PDF files found in the directory.")

    if len(file_path) > 100:
        raise ValueError("Too many PDF files found in the directory.")


def process_document(file_path, queries):
    '''
    TODO:
    :param file_path:
    :param queries:
    :param rag_tool:
    :return:
    '''
    documents = load_pdf(file_path)
    vector = create_vectorstore(documents)
    rag_tool = GradeAndGenerateRagTool(vector)

    extracted_info = {}

    for key, query in queries.items():
        # 使用 RAG 工具进行检索和回答生成
        search_results = rag_tool.search_vector(query)

        for result in search_results:
            text = result["text"]
            # 评分文档与问题的相关性
            relevance_score = rag_tool.grade(query, text)

            if relevance_score == "yes":
                # 生成答案
                answer = rag_tool.generate(query, text)

                # 检查生成的回答是否基于文档
                hallucination_check = rag_tool.hallucinations(text, answer)

                if hallucination_check == "no":
                    # 检查回答是否解决了问题
                    answer_quality = rag_tool.answer_question(query, answer)

                    if answer_quality == "yes":
                        extracted_info[key] = answer
                        break

    return extracted_info


def create_output_directory(pdf_path):
    """ 创建输出目录。"""
    # 获取 PDF 文件所在的目录，并创建一个名为 'output' 的子目录
    output_dir = os.path.join(os.path.dirname(pdf_path), 'output')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"Directory created or exists at: {output_dir}")
    return output_dir


def write_to_csv(extracted_info, file_path, keys, seq_num, output_dir):
    """
    将提取的信息写入CSV文件，并确保CSV文件位于输出目录中。

    :param extracted_info: 提取的信息字典
    :param file_path: CSV文件的基本名称
    :param keys: 要写入的键列表
    :param seq_num: 序号
    :param output_dir: 输出目录
    """
    # 构建完整的CSV文件路径
    csv_file_path = os.path.join(output_dir, file_path)

    row_data = [seq_num] + [extracted_info.get(key, '') for key in keys]
    df = pd.DataFrame([row_data], columns=['Seq'] + keys)
    df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))


def main():
    # PDF文件的主路径
    directory = './ShuLiYou'

    # 查找出所有PDF文件
    file_path_list = find_pdf_files(directory)

    # 验证PDF文件数量是否不为0且在100个以内。
    validate_pdf_file(file_path_list)

    # 定义查询摘要、正文等
    queries = {
        "title": "提取论文标题",
        "label": "提取论文的标签,或者关键词",
        "authors": "提取论文的作者。如:Juanjuan Wang,Hong Lu,Yi He,Chunxiu Jing,Hao Wei*",
        "journal": "提取论文所属的期刊。如: Journal of the American Chemical Society",
        "year": "提取论文发表年份。如:2022",
        "doi": "提取论文的DOI。如: 10.1021/jacs.2c10570",
        "vol": "提取论文的卷,如: 144",
        "issue": "提取论文的期,如:第49期",
        "pages": "提取论文的页码。如:22433-22439",

        "abstract": "提取文档的摘要。提取文献的概要内容，如发现、策略、结果、解决了什么问题",

        "project_development": "课题的发展历程和示例，本工作研究目的或推论的提出，可能会提出创新点，如应用什么新策略，解决了什么问题。",
        "reaction": "反应优化（或条件筛选），描述哪些因素对反应的哪些结果产生影响。如催化剂、配体、添加剂、溶剂、温度、时间、空气、水等，对反应产率或选择性产生哪些影响",
        "sub_range": "底物范围，描述反应对各类底物的适用性，官能团耐受性，不同底物对产率、选择性的影响。通常会增加生物活性分子、药物分子或其衍生物的反应示例，以表明潜在应用性",
        "potential_app_reaction": "反应的潜在应用研究，展示应用示例。通常展示诸如反应量级（如克级）放大后的结果、简化了某类重要分子的合成、制备多样分子的可能性等",
        "mechanism_study": "机理研究，机理的推导和研究的逻辑、结果。如进行了什么实验，为了验证什么问题，实验结果如何，得到什么结论",
        "mechanism_study_conclusion": "机理研究的结论，描述反应可能经历的历程。如反应如何引发，产生了哪些关键中间体，是否有实验证明了该推论。",

        "summary": "提取文档的总结。",
    }

    # 处理PDF文件
    for seq_num, file_path in enumerate(file_path_list, start=1):

        # 创建输出目录
        output_dir = create_output_directory(file_path)

        extracted_info = process_document(file_path, queries)

        # 写入参考信息
        ref_info_keys = ['title', 'label', 'authors', 'journal', 'year', 'doi', 'vol', 'issue', 'pages']
        write_to_csv(extracted_info, 'ref_info.csv', ref_info_keys, seq_num, output_dir)

        # 写入摘要信息
        write_to_csv(extracted_info, 'abstract.csv', ['abstract'], seq_num, output_dir)
        write_to_csv(extracted_info, 'project_development.csv', ['project_development'], seq_num, output_dir)
        write_to_csv(extracted_info, 'reaction.csv', ['reaction'], seq_num, output_dir)
        write_to_csv(extracted_info, 'sub_range.csv', ['sub_range'], seq_num, output_dir)
        write_to_csv(extracted_info, 'potential_app_reaction.csv', ['potential_app_reaction'], seq_num, output_dir)
        write_to_csv(extracted_info, 'mechanism_study.csv', ['mechanism_study'], seq_num, output_dir)
        write_to_csv(extracted_info, 'mechanism_study_conclusion.csv', ['mechanism_study_conclusion'], seq_num, output_dir)
        write_to_csv(extracted_info, 'summary.csv', ['summary'], seq_num, output_dir)

        # 输出提取的信息
        for key, value in extracted_info.items():
            print(f"{key}: {value}")
            print()


if __name__ == "__main__":
    main()
