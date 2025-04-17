import streamlit as st
import re
import os
import tempfile
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
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

def calculate_financial_ratios(debt, equity, receivables, inventories, payables, current_assets, current_liabilities, total_assets, net_income, revenue):
    # Liquidity Ratios
    current_ratio = current_assets / current_liabilities if current_liabilities else float('inf')
    quick_ratio = (current_assets - inventories) / current_liabilities if current_liabilities else float('inf')

    # Profitability Ratios
    net_profit_margin = (net_income / revenue) * 100 if revenue != 0 else 0
    return_on_equity = (net_income / equity) * 100 if equity != 0 else 0

    # Leverage Ratios
    debt_to_equity = debt / equity if equity != 0 else float('inf')

    # Earnings per Share
    eps = net_income / 1_000_000  # Assuming a million shares outstanding for simplicity

    return {
        "Debt to Equity Ratio": debt_to_equity,
        "Net Profit Margin (%)": net_profit_margin,
        "Return on Equity (%)": return_on_equity,
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Earnings per Share (EPS)": eps
    }

def visualize_ratios(ratios):
    # Bar chart for key ratios
    ratio_labels = list(ratios.keys())
    ratio_values = list(ratios.values())

    fig, ax = plt.subplots()
    ax.barh(ratio_labels, ratio_values, color=['green', 'blue', 'orange', 'red', 'purple', 'yellow'])
    ax.set_xlabel('Ratio Value')
    ax.set_title('Key Financial Ratios')

    st.pyplot(fig)

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
                net_income = safe_get("What is the net income in USD?", agent)
                revenue = safe_get("What is the total revenue in USD?", agent)
            with col2:
                account_receivables = safe_get("What are the account receivables in the financial statement?", agent)
                inventories = safe_get("What is the inventory value mentioned in the financial statement?", agent)
                account_payables = safe_get("What are the account payables mentioned in the financial statement?", agent)
                current_liabilities = safe_get("What is the total value of current liabilities in USD?", agent)



            st.subheader("📊 Key Financial Ratios")

            # Show intermediate inputs for transparency/debugging
            inputs_dict = {
                "Total Debt": total_debt,
                "Total Equity": total_equity,
                "Accounts Receivable": account_receivables,
                "Inventories": inventories,
                "Accounts Payable": account_payables,
                "Current Assets": current_assets,
                "Current Liabilities": current_liabilities,
                "Total Assets": total_assets,
                "Net Income": net_income,
                "Revenue": revenue
            }

            st.write("🧾 **Inputs to Ratio Calculation:**")
            st.json(inputs_dict)

            # Show Ratios
            st.subheader("📊 Key Financial Ratios")
            ratios = calculate_financial_ratios(
                total_debt, total_equity, account_receivables, inventories,
                account_payables, current_assets, current_liabilities, total_assets, net_income, revenue
            )

            for ratio, value in ratios.items():
                st.metric(
                    label=f"{ratio}",
                    value=f"{value:.2f}" if isinstance(value, float) else value
                )

            # Visualize Ratios
            st.subheader("📉 Financial Ratios Visualization")
            visualize_ratios(ratios)

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

# Ask Question via Audio or Text
st.subheader("🎤 Ask a Question by Voice or Text")
audio_data = mic_recorder()
text_query = st.text_input("Or enter your question manually:")

query = None

if audio_data is not None:
    # Debugging: Check the type and structure of audio_data
    st.write(f"Audio data type: {type(audio_data)}")
    
    # If it's a dictionary, inspect its contents
    if isinstance(audio_data, dict):
        st.write(f"Audio data contents: {audio_data}")  # See the dictionary structure
        
        # Extract the audio file path (adjust based on the actual structure)
        audio_file_path = audio_data.get("file_path")  # Assuming the dictionary contains a file_path key
        
        if audio_file_path:
            with sr.AudioFile(audio_file_path) as source:
                r = sr.Recognizer()
                audio = r.record(source)
                try:
                    query = r.recognize_google(audio)
                    st.success(f"🗣️ Transcribed Question: {query}")
                except sr.UnknownValueError:
                    st.error("Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
        else:
            st.error("No audio file path found in the received data.")
    else:
        st.error(f"Received audio data is not in the expected dictionary format. It's of type {type(audio_data)}")

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

        # Cleanup audio file
        os.unlink(fp.name)

    # Cleanup PDF
    os.unlink(pdf_path)

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
