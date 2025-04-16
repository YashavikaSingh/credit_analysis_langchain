import streamlit as st
import re
import os
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Hardcoded API key (replace with your actual API key)
OPENAI_API_KEY = "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"

st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")
st.title("Financial Statement Analyzer")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Financial Statement PDF", type="pdf")

def safe_get(query, agent):
    """Safely extract financial data from a query"""
    response = agent.run({"question": query, "chat_history": []})
    # st.write(f"**Query:** {query}")
    # st.write(f"**Response:** {response}")

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
    """Calculate key financial ratios"""
    # Avoid division by zero
    if equity == 0:
        debt_to_equity = float('inf')
        equity_ratio = 0
    else:
        debt_to_equity = debt / equity
        equity_ratio = equity / total_assets
        
    # Net Working Capital
    net_working_capital = receivables + inventories - payables
    
    # Current Ratio
    if current_liabilities == 0:
        current_ratio = float('inf')
    else:
        current_ratio = current_assets / current_liabilities
    
    return {
        "Debt to Equity Ratio": debt_to_equity,
        "Net Working Capital": net_working_capital,
        "Equity Ratio": equity_ratio,
        "Current Ratio": current_ratio
    }

if uploaded_file:
    try:
        with st.spinner("Processing financial statement..."):
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            
            # Initialize LangChain components
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            
            # Load document
            pdf_loader = PyMuPDFLoader(pdf_path)
            documents = pdf_loader.load()
            
            # Create vector store
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            
            # Create agent
            agent = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
            
            # Extract financial data
            st.subheader("Extracting Financial Data")
            
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
            
            # Calculate ratios
            ratios = calculate_financial_ratios(
                total_debt, total_equity, account_receivables, inventories, 
                account_payables, current_assets, current_liabilities, total_assets
            )
            
            # Display results
            st.subheader("Financial Ratios")
            
            # Create a color-coded financial health assessment
            def get_ratio_status(ratio_name, value):
                if ratio_name == "Debt to Equity Ratio":
                    return "🟢 Good" if value < 2 else "🟠 Warning" if value < 3 else "🔴 Concern"
                elif ratio_name == "Current Ratio":
                    return "🟢 Good" if value > 1.5 else "🟠 Warning" if value > 1 else "🔴 Concern"
                elif ratio_name == "Equity Ratio":
                    return "🟢 Good" if value > 0.5 else "🟠 Warning" if value > 0.3 else "🔴 Concern"
                else:
                    return "ℹ️ Informational"
            
            # Display ratios in a nice format
            for ratio, value in ratios.items():
                status = get_ratio_status(ratio, value)
                st.metric(
                    label=f"{ratio} - {status}", 
                    value=f"{value:.2f}",
                )
            
            # Custom financial analysis
            st.subheader("Financial Analysis")
            custom_query = st.text_input("Ask a question about this financial statement:")
            if custom_query:
                response = agent.run({"question": custom_query, "chat_history": []})
                st.write(response)
                
            # Clean up temporary file
            os.unlink(pdf_path)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a financial statement PDF to begin analysis.")
    
    # Example section
    with st.expander("How to use this app"):
        st.write("""
        1. Upload a financial statement PDF
        2. The app will extract key financial data and calculate important ratios
        3. You can also ask custom questions about the financial statement
        
        This app uses LangChain and OpenAI to analyze financial statements and extract key metrics.
        """)