import os

proxy_hosts = {"localhost", "127.0.0.1", "::1"}
proxy_hosts.update(
    host.strip()
    for host in os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",")
    if host.strip()
)
os.environ["NO_PROXY"] = ",".join(sorted(proxy_hosts))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Set offline mode for HuggingFace if necessary
os.environ["HF_HUB_OFFLINE"] = "1"


def get_ollama_llm():

    # Initializes the local ChatOllama LLM 

    print("sending to Local LLM ...")

    return ChatOllama(
        model="gemma3:4b",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,  # Low temperature for deterministic RAG answers
       
    )


def build_rag_pipeline(retriever, include_context=False):
    """
    Constructs an LCEL (LangChain Expression Language) RAG pipeline connecting 
    the retriever to a locally hosted Ollama LLM via ChatOllama.
    """

    # Optimized persian prompt template for concise Persian textbook Q&A
    prompt_template = """
تو پاسخ‌گوی دقیق پرسش‌ها هستی. پرسش را فقط با استفاده از «متن بازیابی‌شده» پاسخ بده.

قواعد:
- پاسخ را به فارسی و مستقیم بنویس؛ فقط اطلاعاتی را بیاور که برای پاسخ به پرسش لازم است.
- متن بازیابی‌شده تنها منبع مجاز است. از دانش بیرونی، حدس یا تکمیل اطلاعات استفاده نکن.
- اگر پرسش چند بخش دارد، به تک‌تک بخش‌ها پاسخ بده و اطلاعات مرتبط را از بخش‌های مختلف متن کنار هم بگذار.
- اگر برای بخشی از پاسخ اطلاعات کافی نیست، همان بخش را مشخص کن و بنویس: «اطلاعات کافی در متن موجود نیست.»
- اگر متن هیچ‌کدام از بخش‌های پرسش را پاسخ نمی‌دهد، فقط بنویس: «اطلاعات کافی در متن موجود نیست.»
- اگر پاسخ به‌صورت صریح یا با بازنویسی روشن در متن آمده، همان پاسخ را بده؛ دربارهٔ نبود اطلاعات توضیح اضافه نده.
- عدد، فرمول، نام یا رابطهٔ علت‌ومعلولی را فقط وقتی بیاور که متن از آن پشتیبانی می‌کند.
- متن بازیابی‌شده را خلاصه یا بازنویسی نکن، مگر به‌اندازه‌ای که برای پاسخ لازم باشد.
- استدلال درونی، مراحل فکر، مقدمه، نتیجه‌گیری اضافه، جدول و عبارت‌هایی مثل «طبق متن» ننویس.
- فقط پاسخ نهایی را برگردان.

متن بازیابی‌شده:
{context}

پرسش:
{question}

پاسخ:
"""
  

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        """Merges retrieved document chunks into a single clean context string."""
        if not docs:
            return "NO CONTEXT RETRIEVED"
        return "\n\n".join(doc.page_content for doc in docs)

    # Instantiate optimized ChatOllama model
    llm = get_ollama_llm()

    # Build context and question inputs for LCEL chain
    rag_input = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    
    # LCEL pipeline: Prompt -> ChatOllama -> StrOutputParser
    answer_chain = prompt | llm | StrOutputParser()

    # Return both context and generated answer when evaluation mode is active
    if include_context:
        return rag_input | RunnablePassthrough.assign(answer=answer_chain)

    return rag_input | answer_chain
