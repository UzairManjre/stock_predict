param([string]$Docx, [string]$Pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false; $w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open($Docx, $false, $false, $false)
  $d.TablesOfContents.Item(1).Update()
  $d.Save()
  "pages=$($d.ComputeStatistics(2)) words=$($d.ComputeStatistics(0))"
  $d.SaveAs2($Pdf, 17)
  "pdf saved"
  $d.Close(0)
} catch { "ERROR: $($_.Exception.Message)" } finally { $w.Quit() }
