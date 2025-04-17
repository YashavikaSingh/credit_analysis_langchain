import streamlit as st
import re
import os
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Set page settings
st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")
st.title("📊 Financial Statement Analyzer")

# Load API key
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Sidebar - Upload
st.sidebar.header("📁 Upload Statement")
uploaded_file = st.sidebar.file_uploader("Upload Financial Statement PDF", type="pdf")

# Utility functions
def safe_get(query, agent):
    response = agent.run({"question": query, "chat_history": []})
    match = re.search(r"\$?[\d,]+\.\d+|\$?[\d,]+", response)
    if match:
        numeric_str = match.group(0).replace(",", "").replace("$", "")
        try:
            return float(numeric_str)
        except ValueError:
            st.warning(f"Could not convert extracted number to float: {numeric_str}")
            return 0.0
    else:
        st.warning(f"No numeric value found for query: {query}")
        return 0.0

def calculate_financial_ratios(debt, equity, receivables, inventories, payables, current_assets, current_liabilities, total_assets):
    if equity == 0:
        debt_to_equity = float('inf')
        equity_ratio = 0
    else:
        debt_to_equity = debt / equity
        equity_ratio = equity / total_assets

    net_working_capital = receivables + inventories - payables
    current_ratio = current_assets / current_liabilities if current_liabilities else float('inf')

    return {
        "Debt to Equity Ratio": debt_to_equity,
        "Net Working Capital": net_working_capital,
        "Equity Ratio": equity_ratio,
        "Current Ratio": current_ratio
    }

def get_ratio_status(ratio_name, value):
    if ratio_name == "Debt to Equity Ratio":
        return "🟢 Good" if value < 2 else "🟠 Warning" if value < 3 else "🔴 Concern"
    elif ratio_name == "Current Ratio":
        return "🟢 Good" if value > 1.5 else "🟠 Warning" if value > 1 else "🔴 Concern"
    elif ratio_name == "Equity Ratio":
        return "🟢 Good" if value > 0.5 else "🟠 Warning" if value > 0.3 else "🔴 Concern"
    else:
        return "ℹ️ Informational"

# Main app logic
if uploaded_file:
    try:
        with st.spinner("🔍 Processing financial statement..."):
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name

            # LangChain setup
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            pdf_loader = PyMuPDFLoader(pdf_path)
            documents = pdf_loader.load()
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            agent = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

            # Financial extraction
            st.subheader("📥 Extracting Financial Data")
            col1, col2 = st.columns(2)
            with col1:
                total_assets = safe_get("What is the total value of assets in USD?", agent)
                total_debt = safe_get("What is the total debt mentioned in the financial statement?", agent)
                total_equity = safe_get("What is the total equity mentioned in the financial statement?", agent)
                current_assets = safe_get("What is the total value of current assets in USD?", agent)
            with col2:
                account_receivables = safe_get("What are the account receivables in the financial statement?", agent)
                inventories = safe_get("What is the inventory value mentioned in the financial statement?", agent)
                account_payables = safe_get("What are the account payables mentioned in the financial statement?", agent)
                current_liabilities = safe_get("What is the total value of current liabilities in USD?", agent)

            # Display ratios
            st.subheader("📊 Key Financial Ratios")
            ratios = calculate_financial_ratios(
                total_debt, total_equity, account_receivables, inventories,
                account_payables, current_assets, current_liabilities, total_assets
            )

            for ratio, value in ratios.items():
                st.metric(
                    label=f"{ratio} - {get_ratio_status(ratio, value)}",
                    value=f"{value:.2f}" if isinstance(value, float) else value
                )

            # Custom Query
            st.subheader("🤖 Ask a Question About the Financials")
            custom_query = st.text_input("Enter your question:")
            if custom_query:
                response = agent.run({"question": custom_query, "chat_history": []})
                st.write(response)

            # Cleanup
            os.unlink(pdf_path)

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

else:
    st.info("📤 Please upload a financial statement PDF to begin.")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        1. Upload a financial statement (PDF).
        2. The app extracts financial metrics using OpenAI + LangChain.
        3. Key financial ratios are calculated and displayed.
        4. You can also ask questions about the content.

        _Built with Streamlit, LangChain, and OpenAI._
        """)
