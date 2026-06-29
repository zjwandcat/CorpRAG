try {
    $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:8001/')
    $req.Timeout = 10000
    $req.Method = 'GET'
    $resp = $req.GetResponse()
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $content = $sr.ReadToEnd()
    $sr.Close()
    Write-Output ("Root page status: " + $resp.StatusCode)
    Write-Output ("Content length: " + $content.Length)
} catch {
    Write-Output ("Error accessing /: " + $_.Exception.Message)
}

Write-Output "---"

try {
    $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:8001/api/v1/config/apikey')
    $req.Timeout = 10000
    $req.Method = 'GET'
    $resp = $req.GetResponse()
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $content = $sr.ReadToEnd()
    $sr.Close()
    Write-Output ("API Key status: " + $content)
} catch {
    Write-Output ("Error accessing /api/v1/config/apikey: " + $_.Exception.Message)
}