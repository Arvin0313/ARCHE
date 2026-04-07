import os
import json
import time
import openai
from typing import List, Dict, Any, Optional

# 配置Deepseek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # 默认模型，可根据需要改为 deepseek-reasoner

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

def call_deepseek_api(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: int = 120,
    max_retries: int = 3,
    retry_delay: int = 5
) -> str:
    """
    调用Deepseek API，带有重试机制
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Deepseek API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"将在{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("达到最大重试次数，放弃请求")
                raise

def extract_json_from_response(response_text: str) -> Any:
    """
    从响应文本中提取JSON对象或数组
    """
    import re
    # 去除可能的markdown代码块标记
    cleaned_text = re.sub(r"```(?:json)?", "", response_text, flags=re.IGNORECASE).strip()
    
    # 尝试匹配JSON对象或数组
    json_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始文本: {match.group(0)[:200]}...")
    
    # 如果没有找到有效的JSON，返回原始文本
    return response_text

def safe_json_call(
    messages: List[Dict[str, str]],
    expect_array: bool = False,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> Any:
    """
    安全的JSON调用：调用API并尝试解析JSON响应
    返回解析后的JSON对象/数组，或原始文本
    """
    response = call_deepseek_api(messages, model=model, **kwargs)
    parsed = extract_json_from_response(response)
    
    if expect_array and not isinstance(parsed, list):
        print("警告：期望数组但未得到数组")
        return []
    elif not expect_array and isinstance(parsed, list) and len(parsed) == 1:
        # 如果期望对象但得到单元素数组，返回数组中的元素
        return parsed[0]
    
    return parsed

# 工具函数：分割查询字符串（按<>分割）
def split_queries(text: str) -> List[str]:
    """
    将模型生成的queries字符串按 <> 分割，并去除多余空格
    """
    parts = text.split('<>')
    queries = [p.strip() for p in parts if p.strip()]
    return queries

# 工具函数：保存模型输出到JSON文件
def save_model_output(
    raw_output: List[Dict], 
    filepath: str, 
    parse_hypotheses: bool = False
) -> None:
    """
    处理模型生成的列表并保存为 JSON 文件
    """
    import re
    
    processed: List[Dict[str, Any]] = []
    
    for item in raw_output:
        new_item = dict(item)  # 拷贝，避免修改原始数据
        
        if parse_hypotheses and isinstance(new_item.get("hypotheses"), str):
            # 去掉 ```json ... ``` 标记
            hypotheses_str = re.sub(r"```json|```", "", new_item["hypotheses"]).strip()
            
            # 尝试解析为 JSON
            try:
                new_item["hypotheses"] = json.loads(hypotheses_str)
            except json.JSONDecodeError:
                # 如果解析失败，就保持为原始字符串
                new_item["hypotheses"] = hypotheses_str
        
        processed.append(new_item)
    
    # 保存到文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 模型输出已成功保存为 {filepath}")