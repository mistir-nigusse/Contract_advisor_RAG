import PyPDF2

class MyPDF:
    def __init__(self, pdf):
        self.pdf = pdf

    def get_pdf_text(self):
        pdf_reader = PyPDF2.PdfReader(self.pdf)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text() 
        return text