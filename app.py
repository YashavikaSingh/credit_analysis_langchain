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

def create_csv_from_data(data_dict):
    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(data_dict)
    
    # Create a CSV buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer.getvalue()

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
    eps = net_income / 1_000_000  # Assuming a million shares outstanding

    # New Ratios
    net_working_capital = receivables + inventories - payables
    equity_ratio = equity / total_assets if total_assets != 0 else 0

    return {
        "Debt to Equity Ratio": debt_to_equity,
        "Net Profit Margin (%)": net_profit_margin,
        "Return on Equity (%)": return_on_equity,
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Earnings per Share (EPS)": eps,
        "Net Working Capital": net_working_capital,
        "Equity Ratio": equity_ratio
    }

def visualize_ratios_plotly(ratios):
    selected_keys = ["Quick Ratio", "Current Ratio", "Equity Ratio", "Debt to Equity Ratio"]
    visual_ratios = {k: ratios[k] for k in selected_keys if k in ratios}

    ratio_labels = list(visual_ratios.keys())
    ratio_values = list(visual_ratios.values())

    fig = go.Figure(go.Bar(
        x=ratio_values,
        y=ratio_labels,
        orientation='h',
        marker=dict(color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'])
    ))
    fig.update_layout(
        title="Selected Financial Ratios",
        xaxis_title="Ratio Value",
        yaxis_title="Ratio Type",
        showlegend=False
    )
    return fig


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

            data_for_csv = inputs_dict.copy()
            data_for_csv.update(ratios)  # Add ratios to the data

            # Create the CSV
            csv_data = create_csv_from_data(data_for_csv)

            # Provide the user a button to download the CSV
            st.download_button(
                label="📥 Download Financial Data & Ratios as CSV",
                data=csv_data,
                file_name="financial_data_ratios.csv",
                mime="text/csv"
            )

            # Visualize Ratios
            st.subheader("📉 Financial Ratios Visualization")

            fig = visualize_ratios_plotly(ratios)
            st.plotly_chart(fig)

            # Provide the user a button to download the figure as a PNG
            st.download_button(
                label="📥 Download Interactive Chart as PNG",
                data=fig.to_image(format="png"),
                file_name="interactive_ratios_chart.png",
                mime="image/png"
            )


    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

    # Generate a downloadable PDF summary
    st.subheader("📄 Download Summary as PDF")


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

# --- Question Answering Section ---
st.subheader("💬 Ask a Question")
question_type = st.radio("How would you like to ask your question?", ["Text", "Voice"])

# Handle Text Question
if question_type == "Text":
    question = st.text_input("Type your question here:")
    if question:
        response = agent.run({"question": question, "chat_history": []})
        st.write(f"🔍 **Answer:** {response}")
        tts = gTTS(response)
        tts.save("answer.mp3")
        st.audio("answer.mp3")

# Handle Voice Question
elif question_type == "Voice":
    mic = mic_recorder()
    st.write("🔴 Press the button and ask your question!")
    if mic.recorded_audio:
        audio_file = mic.recorded_audio
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file)

        # Use Speech Recognition to convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
            try:
                question_text = recognizer.recognize_google(audio)
                st.write(f"📝 You asked: {question_text}")
                response = agent.run({"question": question_text, "chat_history": []})
                st.write(f"🔍 **Answer:** {response}")
                tts = gTTS(response)
                tts.save("answer.mp3")
                st.audio("answer.mp3")
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
            except sr.RequestError:
                st.error("Could not request results from Google Speech Recognition service.")
