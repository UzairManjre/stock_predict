# Building the project report

`../AlphaEdge_Project_Report.docx` (and the PDF) are generated from these files.

```powershell
cd report\source
npm install                                                    # docx (docx-js)
..\..\venv\Scripts\python.exe make_figures.py figs             # redraw the six figures
node build_report.js ..\AlphaEdge_Project_Report.docx          # write the Word file
powershell -ExecutionPolicy Bypass -File word_finish.ps1 -Docx "$PWD\..\AlphaEdge_Project_Report.docx" -Pdf "$PWD\..\AlphaEdge_Project_Report.pdf"
```

`word_finish.ps1` needs Microsoft Word: it fills in the table of contents and saves the PDF.
Without Word, open the .docx, right-click the table of contents, choose **Update Field**, and save as PDF.
