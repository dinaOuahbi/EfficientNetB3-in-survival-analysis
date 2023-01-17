from fpdf import FPDF
import os
title = 'Classification cnn1 _ classes distribution'

class PDF(FPDF):
    def header(self):
        # logo
        #self.image('/work/shared/ptbc/CNN_Pancreas_V2/Donnees/target.tif', 10, 8, 25)
        #set_font
        self.set_font('helvetica','B', 15)
        #title
        title_w = self.get_string_width(title) + 6
        doc_w = self.w
        self.set_x((doc_w - title_w) / 2)
        #padding
        self.cell(80)
        # title
        self.cell(100,10, 'CNN2_ EfficientNetB3', border=True, ln=True, align='C')
        pdf.cell(50, 10, "c1/c2 : Normal | c3/c4 : Stroma | c5/c6 : Tumor | c7/c8 : Duodenum", ln=True, border=False)
        # line break
        self.ln(20)


    def footer(self):
        # set position of the footer
        self.set_y(-15)
        #set font
        self.set_font('helvetica', 'I', 10)
        # page number
        self.cell(0,10,f'Page {self.page_no()} / {{nb}}', align='C')



#path = input('Give me the images path with the last slash please : ')
#output = input('Give the output name please : ')
pdf = PDF('P', 'mm', 'Letter')

#get total page nb
pdf.alias_nb_pages()

# set auto page break
pdf.set_auto_page_break(auto=True, margin=15)

# add page
pdf.add_page()

# specify font
pdf.set_font('helvetica', 'BIU', 10)
pdf.set_text_color(0,0,0)

path = '/work/shared/ptbc/CNN_Pancreas_V2/Donnees/EfficientNet_1/Resultats/LamesCompletes/pie_classes_distribution/'
my_imgs = os.listdir(path)
for img in my_imgs:
    pdf.cell(0,10,img.split('.')[0], ln=True)
    pdf.image(f'{path}{img}', x = 20, y = None, w = 200, h = 200, type = '', link = '')


pdf.output(f'../clf_cnn2_distribution.pdf')
