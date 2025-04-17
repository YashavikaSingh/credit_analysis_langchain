import streamlit as st
import re
import os
import base64
import tempfile
import speech_recognition as sr
from gtts import gTTS
from streamlit_audiorec import st_audiorec
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Set page layout
st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")
st.title("📊 Financial Statement Analyzer")

# API key from secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("📁 Upload Financial Statement (PDF)", type="pdf")

# --- Utility Functions ---

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

# --- Main App ---
if uploaded_file:
    try:
        with st.spinner("🔍 Processing financial statement..."):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name

            # Use OCR-capable loader
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            # Setup LangChain components
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            agent = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

            # Extracting Financial Data
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

            # Show Ratios
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

            # Ask Question via Audio or Text
            st.subheader("🎤 Ask a Question by Voice or Text")
            audio_data = st_audiorec()
            text_query = st.text_input("Or enter your question manually:")

            query = None

            if audio_data is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data)
                    audio_path = f.name

                r = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio = r.record(source)
                    try:
                        query = r.recognize_google(audio)
                        st.success(f"🗣️ Transcribed Question: {query}")
                    except sr.UnknownValueError:
                        st.error("Speech Recognition could not understand audio.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results from Google Speech Recognition service; {e}")

            elif text_query:
                query = text_query

            if query:
                response = agent.run({"question": query, "chat_history": []})
                st.write(response)

                # Text-to-Speech
                tts = gTTS(text=response, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")

            # Cleanup
            os.unlink(pdf_path)

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

else:
    st.info("📤 Please upload a financial statement PDF to begin.")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        1. Upload a financial statement PDF using the sidebar.
        2. The app extracts financial data using AI (OCR-enabled).
        3. Key financial ratios are calculated and color-coded.
        4. You can ask questions using your **voice or text**.
        5. Responses are shown and **spoken back to you**.

        _Built with Streamlit, LangChain, and OpenAI._
        """)
