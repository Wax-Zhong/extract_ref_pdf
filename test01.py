from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import os
import fnmatch
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.self_rag_tool import GradeAndGenerateRagTool

"""
优化方向:
    1.对于某些标准化信息不通过大模型提取,可以代码手动提取
    2.可在大模型基础上进行化学方向上的微调.如对反应优化,底物范围,反应的潜在应用研究等描述进行微调,使大模型对这些方向上的内容更加敏感.
    3.可以使用自省RAG,对query进行重写.
    4.可持久化向量数据库.
"""


def load_pdf(file_path):
    """
    加载PDF文档。

    :param file_path: PDF文件的路径
    :return: 加载的文档列表
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def create_vectorstore(documents, persist_directory="./chroma_db"):
    """
    根据加载的文档创建Chroma向量数据库。

    :param documents: 加载的文档列表
    :param persist_directory: 向量数据库持久化目录
    :return: Chroma向量数据库
    """
    # 加载本地预训练的embedding模型
    embeddings = HuggingFaceEmbeddings(model_name="./mxbai-embed-large-v1", model_kwargs={"device": "cpu"})

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


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
    :param file_path:
    :param queries:
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
            text = result.page_content  # 更改这一行来正确访问文本内容
            # 评分文档与问题的相关性
            relevance_score = rag_tool.grade(query, text)

            if relevance_score == "yes":
                # 生成答案
                answer = rag_tool.generate(query, text)

                # 幻觉检查:检查生成的回答是否基于文档
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
        "title": "Extract the title of the paper",
        "label": "Extract the tags, or keywords, of the paper",
        "authors": "Extract the author of the paper. For example: Juanjuan Wang,Hong Lu,Yi He,Chunxiu Jing,Hao Wei*",
        "journal": "Extract the journal to which the paper belongs. For example: Journal of the American Chemical Society",
        "year": "Extract the year the paper was published. Such as:2022",
        "doi": "Extract the DOI of the paper. Such as: 10.1021/jacs.2c10570",
        "vol": "Extract a volume of papers, such as: 144",
        "issue": "Extract the period of the paper, for example: Issue 49",
        "pages": "Extract the page number of the paper. For example, 22433-22439",

        "abstract": "Extract a summary of the document. Extract the summary content of the literature, such as findings, strategies, results, what problems were solved",

        "project_development": "The development history and examples of the current topic of the paper are extracted, and the proposal of the research purpose or inference of this work may suggest innovative points, such as what new strategies are applied and what problems are solved.",
        "reaction": "Extract the response optimization (or conditional screening) of the current paper, describing which factors have an impact on which outcomes of the response. Such as catalysts, ligands, additives, solvents, temperature, time, air, water, etc. on the reaction yield or selectivity",
        "sub_range": "Extract the substrate range of the current paper, describe the applicability of the reaction to various substrates, functional group tolerance, and the influence of different substrates on yield and selectivity. Examples of reactions of bioactive molecules, drug molecules, or their derivatives are often added to indicate potential applications",
        "potential_app_reaction": "Extract potential applied research responses from current papers and present application examples. Results such as amplification of the reaction magnitude (e.g., gram), simplification of the synthesis of an important class of molecules, and the possibility of preparing a variety of molecules are usually shown",
        "mechanism_study": "Extract the mechanism research of the current paper, the derivation of the mechanism and the logic and results of the research. If what experiments were conducted, what questions were tested, what were the results of the experiments, and what conclusions were drawn",
        "mechanism_study_conclusion": "The conclusions of the mechanism studies in the current papers are extracted and the possible course of the reaction is described. Such as how the reaction is initiated, what key intermediates are produced, and whether there are experiments to prove the inference.",

        "summary": "Extract the summary of the current paper.",
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
        write_to_csv(extracted_info, 'mechanism_study_conclusion.csv', ['mechanism_study_conclusion'], seq_num,
                     output_dir)
        write_to_csv(extracted_info, 'summary.csv', ['summary'], seq_num, output_dir)

        # 输出提取的信息
        for key, value in extracted_info.items():
            print(f"{key}: {value}")
            print()


if __name__ == "__main__":
    main()
