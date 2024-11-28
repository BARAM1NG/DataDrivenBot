# RAG_DataDriven
### Building a Custom Chatbot for Data-Driven Management Strategies

## Project Background

When using ChatGPT, there is a risk of data leakage as input prompts may be utilized for model training. To ensure internal data security, I'm developing an in-house chatbot. For this purpose, I'm leveraging the open-source LLM, Llama 3, to build the chatbot and integrating a Retrieval-Augmented Generation (RAG) technique to extract information from existing PDFs. Additionally, to enable interactive visualization during chatbot conversations, I'm incorporating PyGWalker to facilitate real-time visualizations within the chatbot interface.

## How to Use
1. Clone the repository :
```
git clone https://github.com/BARAM1NG/RAG_DataDriven.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Start the development server:
```
cd app
streamlit run main.py
```

## Key Features
This project has two key features: `Chatbot` and `Visualization`.

## Chatbot

### LLM
- Using a `llama3` model fine-tuned in Korean as the LLM engine
  - [[llama-3-Korean-Bllossom-8B]](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M)

### Langchain / RAG
- Configure the project's basic logic using LangChain
- Utilize the RAG technique to enable responses based on provided files using `rag_chain`
- If no file is provided, generate responses using the default chain
- Build a cache based on code numbers and store it in memory.

## Visualization

### PyGWalker
- Use `PyGWalker` to integrate with the Streamlit application, providing interactive data visualization functionality.  
- Capable of visualizing small datasets (under 10MB) but not suitable for large-scale data visualization.

## GPTs vs Custom Chatbot

### GPTs
- Provide additional explanations based on the given data.
- Deliver answers that are easy to read and understand.
- Offer creative responses.

### Custom Chatbot
- Tend to return content exactly as found in the RAG data.
- Excel at answering questions about the provided materials but struggle with questions requiring creativity.

## Demo Video
- [Demo Video](asset/시연영상.mp4)

