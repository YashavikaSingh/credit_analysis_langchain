import streamlit as st
import re
import io
import pandas as pd
import os
import tempfile
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import OpenAI
from langchain.schema import Document
import plotly.graph_objects as go
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
# API key from secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]


# Set page layout
st.set_page_config(page_title="Financial Statement Analyzer Long Documents", layout="wide")
st.title("📊 Financial Statement Analyzer")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("📁 Upload Financial Statement (PDF)", type="pdf")

# --- Utility Functions ---

import re

def is_table_like(text):
    """
    Checks if the text contains at least 2 lines with 2 or more numeric values per line.
    This is a simple heuristic to detect table-like numeric structures.
    """
    lines = text.splitlines()
    numeric_lines = 0

    for line in lines:
        # Look for at least two numeric patterns in the line
        if len(re.findall(r"\d[\d,\.]*", line)) >= 2:
            numeric_lines += 1
            if numeric_lines >= 2:
                return True
    return False

def contains_financial_keywords(text):
    """
    Checks for the presence of financial terms that often appear in financial tables.
    """
    keywords = [
        'Balance Sheet', 'Income Statement', 'Cash Flow', 'Revenue',
        'Assets', 'Liabilities', 'Equity', 'Net Income', 'Operating Expenses',
        'Gross Profit', 'EBITDA', 'Financial Position'
    ]
    # Case insensitive match
    return any(keyword.lower() in text.lower() for keyword in keywords)

def looks_like_table(text):
    """
    Returns True if the text contains financial keywords or has a table-like numeric structure.
    """
    return contains_financial_keywords(text) or is_table_like(text)

def extract_financial_tables_ai(documents):
    table_like_sections = []

    # Apply early filtering using heuristic checks
    filtered_docs = [
        doc for doc in documents
        if looks_like_table(doc.page_content)
    ]

    # print(f"Filtered from {len(documents)} to {len(filtered_docs)} potential financial sections.")

    for doc in filtered_docs:
        page_text = doc.page_content.strip()


        messages = [
            SystemMessage(
                content="You are a financial analyst AI. Your job is to identify whether this section contains any financial tables, metrics, or structured financial data."
            ),
            HumanMessage(
                content=f"""Does the following text contain a financial table or numeric data such as income statements, balance sheets, ratios, or metrics?

Respond with 'Yes' or 'No'. Then, if yes, provide a cleaned version of just the relevant table-like or numeric portion.

---
{page_text}
---
"""
            )
        ]

        response = llm(messages).content.strip()
        if response.lower().startswith("yes"):
            # Clean and add only the relevant portion
            cleaned = response.split("\n", 1)[1] if "\n" in response else ""
            table_like_sections.append(cleaned)

    # print(f"Extracted {len(table_like_sections)} financial sections.")
    return "\n\n".join(table_like_sections)



def visualize_ratios(ratios):
    selected_keys = ["Quick Ratio", "Current Ratio", "Equity Ratio", "Debt to Equity Ratio"]
    visual_ratios = {k: ratios[k] for k in selected_keys if k in ratios}

    # Creating a bar chart with Plotly
    fig = go.Figure()

    # Add bars to the chart
    fig.add_trace(go.Bar(
        y=list(visual_ratios.keys()), 
        x=list(visual_ratios.values()), 
        orientation='h', 
        marker=dict(color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'])
    ))

    # Update layout for better styling
    fig.update_layout(
        title="Selected Financial Ratios hi hi",
        xaxis_title="Ratio Value",
        yaxis_title="Ratio",
        template="plotly_dark",  # Optional, dark theme
        margin=dict(l=100, r=100, t=40, b=40),
        showlegend=False
    )

    st.plotly_chart(fig)
    return fig  # Return the Plotly figure


def download_summary_csv(inputs: dict, ratios: dict, filename: str = "financial_summary.csv"):
    """
    Creates and offers a CSV file download with both raw inputs and calculated financial ratios.

    Parameters:
        inputs (dict): Dictionary of raw input financial values.
        ratios (dict): Dictionary of calculated financial ratios.
        filename (str): The desired filename for the CSV download.
    """
    buffer = io.StringIO()

    # Convert and write input values
    inputs_df = pd.DataFrame(inputs.items(), columns=["Metric", "Value"])
    buffer.write("Raw Input Values\n")
    inputs_df.to_csv(buffer, index=False)
    buffer.write("\n")

    # Convert and write calculated ratios
    ratios_df = pd.DataFrame(ratios.items(), columns=["Ratio", "Value"])
    buffer.write("Calculated Financial Ratios\n")
    ratios_df.to_csv(buffer, index=False)

    csv_data = buffer.getvalue()

    st.subheader("⬇️ Download Full Financial Summary")
    st.download_button(
        label="Download Summary as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )


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

            # Extract Financial Data from the PDF using AI
            st.subheader("📥 Extracting Financial Data (AI Processed)")
            
            # Extract financial tables using AI
            extracted_tables = extract_financial_tables_ai(documents)
            
            if extracted_tables:
                st.text_area("Extracted Financial Data", value=extracted_tables, height=300)
            else:
                st.write("No financial tables or relevant numeric data found.")


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
            fig = visualize_ratios(ratios)


    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

    # Generate a downloadable PDF summary
    download_summary_csv(inputs_dict, ratios)


else:
    st.info("📤 Please upload a financial statement PDF to begin.")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        1. Upload a financial statement PDF using the sidebar.
        2. The app extracts financial data using AI (OCR-enabled).
        3. Key financial ratios are calculated and color-coded.
        4. Ask questions using text input.

        _Built with Streamlit, LangChain, and OpenAI._
        """)

# --- Text Question Answering Section ---
st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here:")
if question:
    response = agent.run({"question": question, "chat_history": []})
    st.write(f"🔍 **Answer:** {response}")
