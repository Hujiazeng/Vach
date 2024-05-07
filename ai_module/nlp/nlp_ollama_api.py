import json
import requests
import time

ollama_ip = "127.0.0.1"
ollama_model = "qwen:latest"
def question(cont):
    url=f"http://{ollama_ip}:11434/api/chat"
    req = json.dumps({
        "model": ollama_model,
        "messages": [{
            "role": "user",
            "content": cont
        }], 
        "stream": False
        })
    headers = {'content-type': 'application/json'}
    session = requests.Session()    
    starttime = time.time()
     
    try:
        response = session.post(url, data=req, headers=headers)
        response.raise_for_status()  # 检查响应状态码是否为200

        result = json.loads(response.text)
        response_text = result["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        response_text = "抱歉，我现在太忙了，休息一会，请稍后再试。"
    print("接口调用耗时 :" + str(time.time() - starttime))
    return response_text.strip()

if __name__ == "__main__":
    for i in range(1):
        query = "穷游外国, 有什么推荐的国家"
        response = question(query)
        print("\n The result is ", response)
