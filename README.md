# 📊 Financial Statement Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-App-blueviolet)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://python.langchain.com/docs/get_started/introduction)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-red)](https://openai.com/api/)
[![Deployed App](https://img.shields.io/badge/Deployed-App-brightgreen)](https://creditanalysislangchain-eqzzptzfhkkxsttkesfzge.streamlit.app/)

This application is a powerful tool for analyzing financial statements. It leverages the capabilities of LangChain and OpenAI to extract key financial data from PDF documents, calculate important financial ratios, and provide insightful analysis.

## ✨ Features

-   **PDF Upload:** Easily upload financial statements in PDF format.
-   **Automated Data Extraction:** Uses LangChain and OpenAI to intelligently extract crucial financial data like total assets, debt, equity, receivables, payables, and more.
-   **Ratio Calculation:** Automatically computes essential financial ratios, including:
    -   Debt to Equity Ratio
    -   Net Working Capital
    -   Equity Ratio
    -   Current Ratio
-   **Financial Health Assessment:** Provides a color-coded assessment (🟢 Good, 🟠 Warning, 🔴 Concern) for each ratio, making it easy to understand the financial health of the company.
-   **Custom Query:** Ask specific questions about the financial statement and get answers powered by OpenAI's language model.
-   **User-Friendly Interface:** Built with Streamlit, offering an intuitive and interactive experience.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   An OpenAI API key

### Installation

1.  Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set your OpenAI API key:

    -   Create a `.streamlit/secrets.toml` file in your project directory.
    -   Add your API key to the file:

        ```toml
        [openai]
        api_key = "YOUR_OPENAI_API_KEY"
        ```

### Running the App

1.  Navigate to the project directory in your terminal.
2.  Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3.  Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## 🛠️ How to Use

1.  **Upload:** In the sidebar, upload a financial statement PDF.
2.  **Process:** The app will process the document, extract data, and calculate ratios.
3.  **Review:** View the extracted data and calculated ratios, along with their financial health assessments.
4.  **Ask:** Use the "Financial Analysis" section to ask custom questions about the financial statement.

## 💡 Example Use Cases

-   **Investment Analysis:** Quickly assess the financial health of potential investments.
-   **Credit Risk Assessment:** Evaluate the creditworthiness of a company.
-   **Financial Reporting:** Gain insights from financial statements more efficiently.
-   **Due Diligence:** Streamline the process of analyzing financial data during mergers and acquisitions.

## 📚 Technologies Used

-   **Streamlit:** For building the interactive web application.
-   **LangChain:** For document loading, vector storage, and conversational retrieval.
-   **OpenAI:** For language model capabilities and embeddings.
-   **PyMuPDF:** For PDF document loading.
-   **FAISS:** For efficient similarity search in the vector store.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

MIT

---

**Disclaimer:** This application is intended for informational purposes only and should not be considered financial advice. Always consult with a qualified financial advisor before making any investment decisions.
