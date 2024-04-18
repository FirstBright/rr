from vector_store import Vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
class Chat():

    def ask(self,text):
        vector = Vector_store()
        retriever = vector.process_pdf()
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''

        prompt = ChatPromptTemplate.from_template(template)

        from langchain.callbacks.base import BaseCallbackHandler


        class StreamCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                print(token, end="", flush=True)        
    
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        )


        def format_docs(docs):
            # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
            return "\n\n".join(doc.page_content for doc in docs)


        # 체인을 생성합니다.
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(text)
