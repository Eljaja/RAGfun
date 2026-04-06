from pypdf import PdfReader, PdfWriter

reader = PdfReader("/home/ubuntu/tnovik/main_rag_folder/RAGfun/RAGcircle/test_data/management-center-admin-10-0.pdf")
writer = PdfWriter()
for i in range(77, 123):  # first 5 pages (0-based)
    writer.add_page(reader.pages[i])
with open("test-pages.pdf", "wb") as f:
    writer.write(f)