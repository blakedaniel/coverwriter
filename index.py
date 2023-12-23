import streamlit as st
import pdfplumber
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Load the LLM model
model = AutoModelForCausalLM.from_pretrained('bvanflet/coverletter')
tokenizer = LlamaTokenizer.from_pretrained('bvanflet/coverletter')

def process_pdf(uploaded_file):
    pdf = pdfplumber.open(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def generate_output(company, title, job_description, resume):
    # Combine user input and PDF text into a single prompt
    prompt = f"<s>### PROMPT: ~~COMPANY~~: {company} ~~TITLE~~: {title} ~~JOB DESCIPRTION~~: {job_description} ~~RESUME~~: {resume}  ### RESPONSE: COVER LETTER: "
    input_ids = tokenizer(prompt, return_tensors='pt')
    answer = model.generate(input_ids, max_tokens=600)
    generated_text = tokenizer.decode(answer[:, input_ids.shape[-1]:][0], clean_up_tokenization_spaces=True)
    
    # Check if '###' is in the generated text
    if '###' in generated_text:
        # If '###' is found, truncate the text at that point
        generated_text = generated_text[:generated_text.index('###') + len('###')]
    
    return generated_text

st.title("Cover Letter Generator")

with st.form("Input"):
    company_name = st.text_input("Company Name:")
    job_title = st.text_input("Job Title:")
    job_description = st.text_area("Job Description:")
    resume = st.file_uploader("Upload PDF:", type="pdf")
    submit_button = st.form_submit_button(label="Generate Cover Letter")

if submit_button:
    if resume is not None:
        pdf_text = process_pdf(resume)
    else:
        pdf_text = ""

    output = generate_output(company_name, job_title, job_description, pdf_text)

    st.success("COVER LETTER: \n" + output)
