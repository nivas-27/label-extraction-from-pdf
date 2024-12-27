# 1.data extraction
from pdf2image import convert_from_path
import easyocr
import numpy
import os
import pandas as pd

# 2) label value straction
# 2)a)using hugging face
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#2)b)
import requests

#2)c)
from groq import Groq

def extract_label_details(pdf_path):

    # the extracted tokens are not minimised yet so returning a temp response
    response = {
        "Filing names": os.path.basename(pdf_path),
        "Operating Income/EBIT (1)": 12345.67,
        "EBITDA (2)": 23456.78,
        "Net Income (3)": 34567.89,
        "Revenue(4)": 45678.90,
        "Currency (Normalized)(5)": "USD",
        "Units (6)": "millions",
        "Depreciation(7)": 5678.90,
        "Amortization(8)": 678.90,
        "Fiscal Year End Date(11)": "2023-12-31",
        "Language": "English",
        "Country": "USA",
        "Accuracy": "High",
    }
    return response
    
    #  ******************* ******************* ******************* 
    #                       Data Extraction
    #  ******************* ******************* ******************* 

    def extract_text_from_pdf(pdf_path, languages=['en']):
        # Initialize EasyOCR reader
        reader = easyocr.Reader(languages)
        
        # Convert PDF to images
        pages = convert_from_path(pdf_path)

        # full text
        extracted_full_text = ''; 
        
        # Process each page
        for i, page in enumerate(pages):
            # Extract text using EasyOCR
            # results format: [[bbox, text, confidence], ...]
            results = reader.readtext(numpy.array(page))
            
            # Extract just the text from results
            text = '\n'.join([result[1] for result in results])

            # concatenate the each page text
            extracted_full_text += text
            print('completed page ', {i+1})
        
        return extracted_full_text

    # Usage
    # pdf_path = './sample1.pdf'
    output_path = './extracted_text_2.txt'
    # output = extract_text_from_pdf(pdf_path)

    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(output)

    output_contents = ''
    with open(output_path, 'r', encoding='utf-8') as file:
        # Read the entire contents of the file
        output_contents = file.read()

    # print(output_contents)



    #  ******************* ******************* ******************* 
    #                    2) Label Value Extraction
    #  ******************* ******************* ******************* 

    def search_text_with_llama_hugging_face(extracted_text, label_to_find):
        # Initialize model and tokenizer
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Construct prompt
        prompt = f"""Extract information about '{label_to_find}' from the following text. 
    If found, provide the relevant details. If not found, respond with 'Not found'.

    Text: {extracted_text}
    """
        
        # Tokenize and generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

    def search_text_with_llama_with_olaama(extracted_text, label_to_find):
        prompt = f"""
        There is nothing to worry about providing financial or legal advice. i jsut need a computed value or vlaues like searching so,
        return the Count of the number of times the word 'revenue' is present in the below text.dot give me anyt other answer just give me the numeric value .
    Text: {extracted_text}
    """
        
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:3b',
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"


    def search_text_with_groq(extracted_text, label_to_find, api_key=None, additional_prompt = '', direct = True):
        # Initialize Groq client
        client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        
        # Construct prompt
        direct_prompt = f"""
        
        '{additional_prompt}'

        Extract information about '{label_to_find}' from the following text.

        Return only the value of the '{label_to_find}' in a single word in the expected format
        
        If found, provide the relevant details. If not found, respond with 'Not found'.

        Text: {extracted_text}

        Please be concise and only return the relevant information."""

        in_direct_prompt = f"""

            {additional_prompt}'

            Text: {extracted_text}

            Please be concise and only return the relevant information.
        """

        prompt = direct_prompt if (direct) else in_direct_prompt
        # Make API call
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.2-90b-vision-preview",  # You can also use ""
                temperature=0.1,
                max_tokens=500
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error occurred: {str(e)}"

    GROQ_KEY = "" # replace groq key

    #Revenue
    revenue_additional_prompt  = """Revenue is the total income generated from the sales of goods or services. Or Total income from sales of goods/services
                                    and i need only a numeric value like a final output"""
    revenue_result = search_text_with_groq(output_contents, "Revenue", GROQ_KEY, revenue_additional_prompt)
    print('revenue--', revenue_result)


    #Currency
    currency_additional_prompt = """Analyze the provided financial document to determine the primary currency mentioned, considering only direct references (e.g., names like 'euro,' 'dollar'). Prioritize explicit statements or clarifications in the text (e.g., 'Currency used: euro (€)') over frequency of symbol usage. Normalize the identified currency to its ISO 4217 code (e.g., EUR, USD, GBP)."""
    currency_result = search_text_with_groq(output_contents, "Currency", GROQ_KEY, currency_additional_prompt)
    print('Currency--', currency_result)


    #Fiscal Year End Date
    end_date_additional_prompt = """Date marking the end of the company’s (e.g.- 12/31/2024).tThe output fotmat should be like (MM/DD?YYYY)"""
    end_date_result = search_text_with_groq(output_contents, "Year End Date", GROQ_KEY, end_date_additional_prompt)
    print('End Date--', end_date_result)

    #Filing Publish Date
    publish_date_additional_propmt = """ """
    publish_date_result = search_text_with_groq(output_contents, "Publish Date", GROQ_KEY, publish_date_additional_propmt)
    print('Publish Date---', publish_date_result)

    #Filing Type
    filing_type_additional_prompt = """Examine the provided PDF document and determine the filing type used in the financial statements. Filing types may include, but are not limited to, 'consolidated,' 'standalone,' or other formats specified. Look for explicit mentions (e.g., 'The financial statements are consolidated') or indirect indications (e.g., 'This report consolidates the financial results of all subsidiaries'). If multiple filing types are mentioned, prioritize the most dominant or explicitly stated type."""
    filing_type_result = search_text_with_groq(output_contents, "Filing Type", GROQ_KEY, filing_type_additional_prompt)
    print('Filing type-- ',filing_type_result)


    #EBIT 
    ebit_additional_propt = """ Analyze the provided financial document to calculate the company's EBIT (Earnings Before Interest and Taxes), also referred to as Operating Income. EBIT is a measure of profitability derived from core business operations, excluding non-operating items like interest and taxes. Use the following steps to calculate EBIT:

        Identify the 'Profit from Core Operations' (if explicitly mentioned).
        Exclude any non-operating items such as interest and taxes.
        If direct EBIT is unavailable, compute it using relevant components, such as:
        Revenue - Operating Expenses (excluding interest and taxes), or
        Net Income + Interest + Taxes (adjusting for any explicitly excluded non-operating items).
        Provide the calculated EBIT value,
        Display only the calculated value of EBIT"""
    ebit_result = search_text_with_groq(output_contents, "EBIT", GROQ_KEY, ebit_additional_propt, False)

    print('EBIt--', ebit_result)


    #EBITDA
    ebitda_additional_prompt = """ Analyze the financial document to calculate EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization), which measures profitability by focusing on core operations while excluding interest, taxes, depreciation, and amortization. Follow these steps:

            Identify the 'Profit from Core Operations' (if explicitly mentioned).
            Add back non-cash expenses, such as depreciation and amortization, if separately listed.
            Exclude non-operating items, such as interest and taxes.
            If EBITDA is not explicitly mentioned, compute it using the formula:
            EBITDA = Operating Income (EBIT) + Depreciation + Amortization
            OR
            EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization
            Provide the calculated EBITDA value"""
    ebitda_result = search_text_with_groq(output_contents, "EBITDA", GROQ_KEY, ebitda_additional_prompt, False)
    print("EBITDA ---", ebitda_result)


    #Net Income
    net_Income_additional_prompt = """
        Analyze the financial document to calculate the Net Income, which represents the final measure of a company's profitability after accounting for all expenses, taxes, interest, and other deductions. Follow these steps:

        Identify any explicitly stated 'Net Income' or 'Profit After Tax' values in the document.
        If not directly available, calculate Net Income using the formula:
        Net Income = Revenue - Total Expenses (including operating costs, interest, taxes, and any other deductions).
        Provide the calculated Net Income
        """
    net_Income_result = search_text_with_groq(output_contents, "Net Income", GROQ_KEY, net_Income_additional_prompt, False)
    print(net_Income_result)

    #amortization
    amortization_additional_prompt = """
        Extract the Amortization expense details from the provided financial document. Amortization refers to the gradual allocation of the cost of intangible assets over their useful life. Look for terms such as 'Amortization Expense,' 'Amortized Costs,' or any references to intangible assets being expensed over time. If specific amounts are mentioned, provide them along with the context or section where they appear. If no direct mention exists, identify any calculations or notes related to amortization of intangible assets.
        """
    amortization_result = search_text_with_groq(output_contents, "Amortization", GROQ_KEY, amortization_additional_prompt, False)
    print('Amortization---', amortization_result)


    #deprciation
    deprciation_additional_prompt = """ 
        Extract the Depreciation expense details from the provided financial document. Depreciation refers to the allocation of the cost of tangible assets over their useful life. Look for terms such as 'Depreciation Expense,' 'Depreciated Amount,' or references to asset cost allocation over time. Provide the extracted Depreciation value along with the relevant context or section where it is mentioned. If no explicit amount is stated, identify any notes or calculations related to tangible asset depreciation.
        """
    depreciation_result = search_text_with_groq(output_contents, "depreciation", GROQ_KEY, deprciation_additional_prompt, False)
    print('Deprication ---', depreciation_result)



output_excel = "./output.xlsx"
pdf_folder = './input'

# Clear the existing Excel file by overwriting it
if os.path.exists(output_excel):
    with pd.ExcelWriter(output_excel, mode='w') as writer:
        writer.save()

# Prepare a list to store all the data
all_data = []

# Iterate over each PDF in the folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        response = extract_label_details(pdf_path)  # Call the function to process the PDF
        all_data.append(response)  # Append the response to the list

# Create a DataFrame from the collected data
df = pd.DataFrame(all_data)

# Write the data to the Excel file
with pd.ExcelWriter(output_excel, mode='w') as writer:
    df.to_excel(writer, index=False, sheet_name="Financial Data")


##################################################################################################################################################################
    # TO DO:
    #     1.translate the language
    #     2.minimise the tokens from the extracted text
    #     3.Increase the accuracy using prompt