try {
    $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:8001/docs')
    $req.Timeout = 15000
    $req.Method = 'GET'
    $resp = $req.GetResponse()
    Write-Output ("Status: " + $resp.StatusCode)
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $content = $sr.ReadToEnd()
    $sr.Close()
    Write-Output ("Content length: " + $content.Length)
} catch {
    Write-Output ("Error: " + $_.Exception.Message)
}