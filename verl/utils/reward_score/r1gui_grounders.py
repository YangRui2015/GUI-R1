import re
import json
import asyncio, httpx
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from io import BytesIO
import base64
import random


grounding_urls = ['http://huanz-serv-01.csl.illinois.edu:23333/v1','http://huanz-serv-02.csl.illinois.edu:23333/v1']
headers = {"Content-Type": "application/json"}

ug_clients = [AsyncOpenAI(
        base_url=url,
        api_key='token-abc123',
        http_client=httpx.AsyncClient(http2=True, timeout=httpx.Timeout(30.0, read=60.0, connect=10.0)),
    ) for url in grounding_urls]

# 正则
_OUTER_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ACTION_RE = re.compile(r'"action_type":\s*"(.*?)"')
_INPUT_TEXT_RE = re.compile(r'"value":\s*"(.*?)"')
_TARGET_RE = re.compile(r'"action_target":\s*"(.*?)"')


class UgRouter:
    def __init__(self, clients):
        self.clients = clients
        self.n = len(clients)
        self._lock = asyncio.Lock()
        self._next = 0

    async def acquire(self):
        async with self._lock:
            i = self._next
            self._next = (self._next + 1) % self.n
            return i, self.clients[i]

    async def release(self, i):  # 兼容接口
        return

    async def route(self, coro_factory):
        i, client = await self.acquire()
        return await coro_factory(client)

ug_router = UgRouter(ug_clients)

def _b64_png_from_raw(raw_png_or_pnglike_bytes: bytes) -> str:
    # 最快路径：不做 PIL 解码/重编码，直接 base64
    return base64.b64encode(raw_png_or_pnglike_bytes).decode("utf-8")

def pil_to_b64_forug(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
    return img_b64



async def chat_with_ug(ug_client, image: str, prompt: str):
    response = await ug_client.chat.completions.create(
        model="osunlp/UGround-V1-7B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 假设图像格式为jpeg，如果需要其他格式，可以修改
                            "url": f"data:image/png;base64,{image}"
                        }
                    },
                    {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {prompt}

Answer:"""}
                ]
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content


async def call_uground(client, expression, image_bytes):
    image_bs64=_b64_png_from_raw(image_bytes)
    response = await chat_with_ug(client,
                            image=image_bs64,
                            prompt=expression)

    coordinate = eval(response)
    return coordinate


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def extract_action(content):
    content_answer_match = _ANSWER_TAG_RE.search(content)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = _ACTION_RE.search(content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"

def extract_input_text(content):
    content_answer_match = _ANSWER_TAG_RE.search(content)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = _INPUT_TEXT_RE.search(content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

async def extract_coord(content, image_bytes=None):
    # Try to extract action target and feed them into grounding models to get coordinates
    content_answer_match = _ANSWER_TAG_RE.search(content)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            target_match = _TARGET_RE.search(content_answer)
            if target_match:
                target = target_match.group(1)
                # coord = await call_uground(ug_client, target, image_bytes)
                coord = await ug_router.route(
                    lambda client: call_uground(client, target, image_bytes)
                )
                return coord, True
            else:
                return [0, 0, 0, 0], False
        else:
            target = _TARGET_RE.search(content).group(1)
            if target:
                # coord = await call_uground(ug_client, target, image_bytes)
                coord = await ug_router.route(
                    lambda client: call_uground(client, target, image_bytes)
                )
                return coord, True
            else:
                return [0, 0, 0, 0], False
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    
def r1gui_format_reward(predict_str: str) -> float:
    """
    检查 predict_str 是否符合 <think></think><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
    """
    # 检查 <think> 和 <answer> 的外部结构
    if not _OUTER_FORMAT_RE.match(predict_str):
        return 0.0

    # 提取 <answer> 中的内容
    answer_match = _ANSWER_TAG_RE.search(predict_str)
    if not answer_match:
        return 0.0

    # 提取 <answer> 内的内容并解析为 JSON 格式
    answer_content = answer_match.group(1).strip()
    try:
        # 验证answer_content是否符合这样的格式: {
        #    "action_type": "scroll",
        #    "action_target": "Patch Details View",
        #    "value": "scroll down"
        # }
        json_content = json.loads(answer_content)
        if not isinstance(json_content, dict):
            return 0.0
        elif "action_type" in json_content and "action_target" in json_content and "value" in json_content:
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

async def r1gui_accuracy_reward(predict_str: str, ground_truth: str, image_bytes=None) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        ground_truth=json.loads(ground_truth)
        gt_action=ground_truth['action'].lower()
        gt_bbox=ground_truth['gt_bbox']
        gt_input_text=ground_truth['input_text'].lower()
        gt_scale_x, gpt_scale_y=ground_truth['scale']
        gt_bbox=[coord / gt_scale_x if i%2==0 else coord / gpt_scale_y for i,coord in enumerate(gt_bbox)]

        pred_action=extract_action(predict_str).lower()
        pred_input_text=extract_input_text(predict_str).lower()
        # image = Image.open(BytesIO(image_bytes["bytes"])) 

        if pred_action!=gt_action:
            if random.random()<0.02:
                print(f"Action mismatch: pred_action: {pred_action}, gt_action: {gt_action}")
            return 0.0

        if gt_action in ['click']: # type? 'select'
            pred_ratio,_= await extract_coord(predict_str, image_bytes["bytes"])
            pred_bbox = [pred_ratio[0]/1000, pred_ratio[1]/1000]
        else:
            pred_bbox = [-1.0, -1.0]  

        if random.random()<0.02:
            print(f"pred_action: {pred_action}, gt_action: {gt_action}, pred_bbox: {pred_bbox}, gt_bbox: {gt_bbox}, pred_input_text: {pred_input_text}, gt_input_text: {gt_input_text}")
        
        
        if gt_action in ["click"]:
            if len(gt_bbox)==2:
                if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<0.14**2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_bbox)==4:
                if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        elif gt_action in ['type', 'select','scroll']:
            if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    except Exception as e:
        print(f"Error in accuracy reward calculation: {e}")
        return 0.0
    
async def r1gui_grounders_compute_score(predict_str: str, ground_truth: str, image_bytes=None):
    # fmt = r1gui_format_reward(predict_str)       
    # acc = await r1gui_accuracy_reward(predict_str, ground_truth, image_bytes)
    fmt_task = asyncio.create_task(asyncio.to_thread(r1gui_format_reward, predict_str))
    acc_task = asyncio.create_task(r1gui_accuracy_reward(predict_str, ground_truth, image_bytes))
    fmt, acc = await asyncio.gather(fmt_task, acc_task)
    return {
        "overall": 0.9 * acc + 0.1 * fmt,
        "format": fmt,
        "accuracy": acc,
    }