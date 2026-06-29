$env:TEST_API_KEY = 'test-key'
$env:PROVIDER = 'zhipu'
$env:ZHIPU_API_KEY = 'bea90dbf0e4e47c48f7839aa674bec76.UMIQ6jKihQnlvitm'
$env:ZHIPU_MODEL_NAME = 'glm-4.7-flash'
Set-Location 'C:\Users\27553\Desktop\rag\financial-agent-api'
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
