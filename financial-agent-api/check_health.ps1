try {
    $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:8001/health')
    $req.Timeout = 30000
    $req.Method = 'GET'
    $resp = $req.GetResponse()
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
    Write-Output $sr.ReadToEnd()
    $sr.Close()
} catch {
    Write-Output ("Error: " + $_.Exception.Message)
}