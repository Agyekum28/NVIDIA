# Financial Assistiant for NVIDIA using Tavily Search and Groq LLM
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
from langchain_core.caches import InMemoryCache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
from dotenv import load_dotenv


# Custom CSS for theming

st.set_page_config(page_title="Nvidia Stock Assistant",)

st.markdown(
    """
    
    <style>
    }

    /* Buttons */
    .stButton>button {
        background-color: #000000 !important;  /* black buttons */
        color: #00FF00 !important;             /* green text on buttons */
    }

    
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Set up caching for LLM responses
set_llm_cache(InMemoryCache())
# Load environment variables from .env file
load_dotenv()
# LLM and Search Tool Initialization
llm = ChatGroq(model="llama-3.3-70b-versatile temp=0.2", max_retries=3, timeout=120)
search_tool = TavilySearch()


def generate_insights(company_url, competitors, pdf_text):

    # Perform web search to gather information
    search_query = f"Company: {company_url}, Competitors: {competitors}"
    search_results = search_tool.invoke({"query": search_query})
    messages = [
        SystemMessage(
            "You're a helpful Financial assistant with over 20 years in the stock market evaluating companies to better forecast data."
        ),
        HumanMessage(content=f"""
         PDF Content: {pdf_text}    

        Company Info from Tavily: {search_results} https://www.kaggle.com/datasets/elnazalikarami/nvidia-corporation-stock-historical-quotes, https://www.kaggle.com/datasets/rohanroy/nvdia-stock-historical-stock-price, https://www.kaggle.com/code/mhassansaboor/nvidia-stocks-analysis-2025,

        Competitors: {competitors}
        
        Generate a one-page report including the following sections, in addition to any other relevant insights like recent news, financial data, or market trends, and also relaying a forcasted view of the Nvidia stock going forward - when to buy and when to sell 
        1. Company strategy
        2. Possible competitors or partnerships in this area
        3. Leadership and decision-makers relevant to this area
        Format output in clear sections with bullet points.
        """)
    ]

    model_response = llm.invoke(messages)
    return model_response.content


# UI with Streamlit


from PIL import Image
import random

# Load local logo
logo = Image.open("nvidia_logo.jpeg")

st.sidebar.image(logo, use_container_width=True)





st.sidebar.markdown(
    """
    Nvidia Corporation is an American multinational technology company 
    incorporated in Delaware and based in Santa Clara, California. 
    They are best known for their graphics processing units (GPUs)
    for gaming and professional markets, as well as AI and data center technologies.  
    Their products are used in gaming, AI research, data centers, automotive, and more.
    """
)


st.title("Nvidia Stock Assistant ")
st.subheader("Nvidia Stock Data ")
st.divider()
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Combine PDF text into a single string
    pdf_text = "\n".join([doc.page_content for doc in documents])

    st.session_state["pdf_text"] = pdf_text   # store it for later
    st.success("PDF uploaded successfully!")

st.divider()
stock_ticker = st.text_input("Nvidia Ticker:")
competitors = st.text_input("Competitors (comma-separated):", "AMD, Intel, Qualcomm")

if st.button("Generate Report"):
    if uploaded_file or (stock_ticker):
        with st.spinner("Generating report..."):
            pdf_text = st.session_state.get("pdf_text", "")
            insights = generate_insights(stock_ticker, competitors, pdf_text)
            st.subheader("Stock Insights")
            st.write(insights)
    else:
        st.warning("Please provide Nvidia's stock ticker or upload a PDF document.")

st.divider()


