from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv
import os
from fpdf import FPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize Tavily
tavily = TavilySearch(api_key=TAVILY_API_KEY)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic", "context"],
    template="""
You are a helpful research assistant. Summarize the following search results for "{topic}" clearly and concisely:

{context}

Summary:
"""
)

# LLMs
gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemma-7b-it",  # or "gemini-pro"
    google_api_key=GOOGLE_API_KEY
)

together_client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

def summarize_with_together(topic, context):
    full_prompt = f"""
You are a helpful research assistant. Summarize the following search results for "{topic}" clearly and concisely:

{context}

Summary:
"""
    response = together_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form.get("topic")
        model_choice = request.form.get("model")  # either "gemini" or "together"

        try:
            # 1. Search using Tavily
            results = tavily.invoke(topic)
            search_results = results.get("results", [])
            if not search_results:
                raise Exception("No results found from Tavily.")

            context = "\n\n".join([
                f"{r.get('title', '')}\n{r.get('content', '')}"
                for r in search_results
            ])
            sources = [r.get("url", "") for r in search_results]

            # 2. Use selected model
            if model_choice == "gemini":
                chain = prompt | gemini_llm
                response = chain.invoke({"topic": topic, "context": context})
                summary = response.content
            elif model_choice == "together":
                summary = summarize_with_together(topic, context)
            else:
                raise Exception("Invalid model choice.")

            # 3. Save to PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Research Summary for: {topic}\n\n{summary}")
            pdf.ln(10)
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, "Sources:", ln=True)
            pdf.set_font("Arial", size=11)
            for src in sources:
                pdf.multi_cell(0, 10, src)
            pdf.output("summary.pdf")

            return render_template("index.html", summary=summary, sources=sources, topic=topic)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

@app.route("/download")
def download_pdf():
    return send_file("summary.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
