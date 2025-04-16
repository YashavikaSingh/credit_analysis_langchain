# 📊 Financial Statement Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-App-blueviolet)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://python.langchain.com/docs/get_started/introduction)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-red)](https://openai.com/api/)

This application is a powerful tool for analyzing financial statements. It leverages the capabilities of LangChain and OpenAI to extract key financial data from PDF documents, calculate important financial ratios, and provide insightful analysis.

🔗 **Try it live:** [https://creditanalysislangchain-eqzzptzfhkkxsttkesfzge.streamlit.app/](https://creditanalysislangchain-eqzzptzfhkkxsttkesfzge.streamlit.app/)

## ✨ Features

- **📄 PDF Upload:** Upload financial statements in PDF format.
- **📊 Automated Data Extraction:** Extracts crucial financial data like total assets, debt, equity, receivables, payables, etc.
- **🧮 Ratio Calculation:** Computes:
  - Debt to Equity Ratio
  - Net Working Capital
  - Equity Ratio
  - Current Ratio
- **🟢 Financial Health Assessment:** Color-coded indicators (🟢 Good, 🟠 Warning, 🔴 Concern).
- **🤖 Custom Query:** Ask specific financial questions powered by OpenAI.
- **🖥️ Intuitive Interface:** Built with Streamlit for a smooth user experience.

## 💡 Example Use Cases

- **Investor Analysis:** Understand the financial health of potential investments.
- **Credit Risk Evaluation:** Gauge a company’s ability to pay its obligations.
- **Financial Reporting:** Automate extraction and interpretation of financial data.
- **Due Diligence:** Speed up document review for mergers and acquisitions.

## 📚 Technologies Used

- **Streamlit** – Interactive UI
- **LangChain** – Document parsing, embeddings, retrieval
- **OpenAI** – LLMs for analysis and querying
- **PyMuPDF** – PDF loading
- **FAISS** – Vector similarity search

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to help improve the tool.

## 📄 License

MIT License

---

**Disclaimer:** This tool is for educational and informational purposes only. It should not be considered as financial advice. Please consult a professional for investment or accounting decisions.
