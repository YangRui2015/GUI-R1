import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from datasets import load_dataset
from datasets import Dataset as hf_dataset
from PIL import Image
from io import BytesIO
import sys, time, base64, signal, atexit, threading
import requests
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue as ThreadQueue
import time
import itertools
from openai import OpenAI, AsyncOpenAI
# import requests, json, atexit
# import json, requests
# from io import BytesIO
# from PIL import Image
# from qwen_vl_utils import process_vision_info
import asyncio, httpx


# 模型路径
MODEL_PATH = ""

# 数据路径
DATA_PATH = ""


def _preprocess_to_b64(processor, sample):
    img = Image.open(BytesIO(sample["image"]["bytes"]))
    w, h = img.size
    # # resize image if too large using processor
    # msg = [{"role":"user","content":[
    #     {"type":"image", "image": img},
    #     {"type":"text", "text": "."},
    # ]}]
    # prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs, video_kwargs = process_vision_info(msg, return_video_kwargs=True)
    # inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
    #               padding=True, return_tensors="pt")
    # grid_h, grid_w = inputs["image_grid_thw"][0][1].item(), inputs["image_grid_thw"][0][2].item()
    # patch = int(getattr(processor.image_processor, "patch_size", 14))
    # new_h, new_w = grid_h * patch, grid_w * patch
    # if h > new_h or w > new_w:
    #     img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    # convert BytesIO(sample["image"]["bytes"]) to base64
    b64= base64.b64encode(sample["image"]["bytes"]).decode("utf-8")
    # buf = BytesIO(); img.save(buf, format="PNG")
    # b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    out = dict(sample)
    out["image_b64"] = b64
    out["orig_w"] = w
    out["orig_h"] = h
    out["image"] = ""  # 或 del out["image"]
    out["scale"]=[1.0, 1.0]
    out["image_size"]=[w, h]
    # print(f"Preprocessed image: {w}x{h} -> {new_w}x{new_h}, processor max pixels={processor.image_processor.max_pixels}")
    return out



def _build_payload(sample, model_path):
    instr = sample["instruction"]
    history = sample.get("history", "None")

    # question_description = '''Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {}\n\nInteraction History: {}\n'''
    # system_prompt = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of actions to complete the task. Available actions include 'Click', 'Write', 'Scroll', 'Back', 'LongPress', 'Wait', and 'OpenAPP'."
    prompt_str= f'''You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instr}', with the action history being '{history}'.
            The response should be structured in the following format:
            <think> Your step-by-step thought process here... </think>
            <answer>
            {{
                "action_type": "the type of action to perform, e.g., click, type, select, scroll, complete, close/delete, press_home, press_back, enter",
                "action_target": "the description of the target of the action, such as the color, text, or position on the screen o f the UI element to interact with",
                "value": "the input text or direction ('up', 'down', 'left', 'right') for the action, if applicable; otherwise, use 'no input text'"
            }}
            </answer>
            '''
    messages = [
                {"role":"user", "content":[
                    {"type":"image_url",
                        "image_url":{"url":f"data:image/png;base64,{sample['image_b64']}",
                                    "detail":"high"}},
                    {"type":"text", "text": prompt_str}
                ]}
            ]

    return {
        "model": model_path,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "max_tokens": 1024,
    }

async def _send_one(client, url, headers, payload, meta):
    print(f"Sending to {url} ...")
    r = await client.post(url, headers=headers, content=json.dumps(payload),
                            timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0),)
    if r.status_code != 200:
        # 打印 server 提示，直指问题
        print(f"[{url}] {r.status_code} {r.reason_phrase} -> {r.text[:500]}")
        r.raise_for_status()

    data_json = r.json()
    text = data_json["choices"][0]["message"]["content"]
    out = dict(meta)
    out["pred"] = text
    return out


async def pipeline_send(data_iter, model_path, chat_urls,  
                        headers, ug_client_list=None,
                        cpu_workers=16, queue_size=2048,
                        async_concurrency=256, 
                        results_q=None, max_pixels=2646000):
    """
    data_iter: 可迭代的原始样本
    cpu_workers: Stage1 进程数
    queue_size: 队列容量（防止内存暴涨，提供背压）
    async_concurrency: Stage2 并发度
    """

    limits = httpx.Limits(max_keepalive_connections=512, max_connections=512)
    q = asyncio.Queue(maxsize=queue_size)
    post_q = asyncio.Queue(maxsize=queue_size)
    total = 0

    inflight_total = 0
    inflight_by_url = {u: 0 for u in chat_urls}
    inflight_lock = asyncio.Lock()
    async def inc(url):
        nonlocal inflight_total
        async with inflight_lock:
            inflight_total += 1
            inflight_by_url[url] += 1

    async def dec(url):
        nonlocal inflight_total
        async with inflight_lock:
            inflight_total -= 1
            inflight_by_url[url] -= 1


    async def heartbeat():
        while True:
            await asyncio.sleep(2.0)
            by_url_str = " ".join([f"{k.split(':')[-1]}={v}" for k, v in inflight_by_url.items()])
            print(f"[HB] q={q.qsize()}/{q.maxsize} post_q={post_q.qsize()}/{post_q.maxsize} "
                f"inflight_total={inflight_total} {by_url_str} consumers={num_consumers} grounders={num_grounders}")

    # 1) 生产者：多进程预处理 -> 放入队列
    def produce_batched(executor, it, batch=64):
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
        processor.image_processor.max_pixels = max_pixels
        buf = []
        for s in it:
            buf.append(s)
            if len(buf) >= batch:
                yield [executor.submit(_preprocess_to_b64, processor, x) for x in buf]
                buf.clear()
        if buf:
            yield [executor.submit(_preprocess_to_b64, processor, x) for x in buf]

    # 2) 消费者：从队列取 -> 选择URL -> 发请求 -> 写结果
    async def consumer_loop(client: httpx.AsyncClient, url: str, cid: int):
        print(f"[CONS#{cid}] start for {url}")
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                break
            payload, meta = item
            image_bs64 = meta.pop("image_b64", None)
            try:
                await inc(url)          
                res = await _send_one(client, url, headers, payload, meta)
                if image_bs64 is not None:
                    res["image_b64"] = image_bs64
                await post_q.put(res) # 放入后处理队列
            except Exception as e:
                print(f"[CONS#{cid}@{url}] send error:", repr(e))
            finally:
                await dec(url)   
                q.task_done()

    # 后处理：从 post_q 取 -> 提取动作 -> 调用 UGround（可选） -> 写结果
    async def grounder_loop(gid: int):
        # need_ug_actions = {"click", "clear", "hover", "type"}  # adjust as needed
        while True:
            item = await post_q.get()
            if item is None:
                post_q.task_done()
                break

            try:
                text = item.get("pred", "")
                fields = extract_action_fields(text) or {}
                item["pred_action"] = (fields.get("action_type") or "none").lower()
                item["pred_input_text"] = fields.get("value") or "none"
                # item["pred_action_description"] = fields.get("action_description") or ""
                item["pred_action_target"] = fields.get("action_target") or ""

                action_map = {
                    'longpress': 'long_press',
                    'openapp': 'open_app',
                    'write': 'type',
                    'back': 'press_back'
                }
                if item["pred_action"] in action_map:
                    item["pred_action"] = action_map[item["pred_action"]]
                if item['pred_action'] == 'scroll' and 'pred_action_description' in item:
                    if ' down ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'down'
                    elif ' up ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'up'
                    elif ' left ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'left'
                    elif ' right ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'right'
                    else:
                        item['pred_input_text'] = 'down'  # default to down if not specified
                elif item['pred_action'] == 'swipe' and 'pred_action_description' in item:
                    if ' down ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'up'
                    elif ' up ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'down'
                    elif ' left ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'right'
                    elif ' right ' in item['pred_action_description'].lower():
                        item['pred_input_text'] = 'left'
                    else:
                        item['pred_input_text'] = 'up'  # default to down if not specified

                # If action requires grounding, call UGround
                if item["pred_action_target"] != "" and item["pred_action_target"] != 'None' and ug_client_list:
                    image_b64 = item.get("image_b64")  # was preserved in preprocessing/meta
                    # Prefer instruction; if you want to use action_description instead, switch below
                    description = item["pred_action_target"]
                    ug_xy_0_1000 = await call_uground(ug_client_list[gid%len(ug_client_list)], description, image_b64)  # (x,y) in 0..1000 grid
                    if ug_xy_0_1000 is not None and "orig_w" in item and "orig_h" in item:
                        w, h = item["orig_w"], item["orig_h"]
                        px = int(w * ug_xy_0_1000[0] / 1000.0)
                        py = int(h * ug_xy_0_1000[1] / 1000.0)
                        item["pred_coord"] = [px, py]
                    else:
                        item["pred_coord"] = [-100, -100]
                else:
                    item["pred_coord"] = [-100, -100]

                item.pop("image_b64", None)  # remove to save space
                # Finally emit the enriched record to writer
                if results_q is not None:
                    results_q.put(item)
            except Exception as e:
                print(f"[GROUND#{gid}] error:", repr(e))
            finally:
                post_q.task_done()

    # 3) 并发运行
    default_timeout = httpx.Timeout(connect=20.0, read=300.0, write=300.0, pool=20.0)
    limits = httpx.Limits(max_keepalive_connections=128, max_connections=128)

    # with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
    with ThreadPoolExecutor(max_workers=cpu_workers) as ex:
        # 为每个 URL 创建一个 client
        clients = [(url, httpx.AsyncClient(limits=limits, timeout=default_timeout))
                for url in chat_urls]

        # 每个 URL 分配相同数量的消费者
        per_url_cons = max(1, async_concurrency // max(1, len(clients)))
        consumers = []
        for url, client in clients:
            for i in range(per_url_cons):
                consumers.append(asyncio.create_task(consumer_loop(client, url, i)))

        # 根据真实消费者数量，发送相同数量的哨兵
        num_consumers = len(consumers)
        grounders = [asyncio.create_task(grounder_loop(i)) for i in range(async_concurrency)]
        num_grounders = len(grounders)

        async def producer_loop(executor):
            produced = 0
            for fut_batch in produce_batched(executor, data_iter, batch=16):  # 可把 64 -> 128
                for fut in as_completed(fut_batch):
                    try:
                        s = fut.result()
                    except Exception as e:
                        print("[PROD] preprocess error:", repr(e))
                        continue
                    payload = _build_payload(s, model_path)
                    await q.put((payload, s))
                    produced += 1
            # 投递哨兵（和消费者数量一致）
            for _ in range(num_consumers):
                await q.put(None)
            print(f"[PROD] done, queued {produced} items + {num_consumers} sentinels")

        hb = asyncio.create_task(heartbeat())

        try:
            prod = asyncio.create_task(producer_loop(ex))
            await asyncio.gather(prod)   # 只等生产者完成投递
            await q.join()               # 等队列清空
            for _ in range(num_grounders):
                await post_q.put(None)
            await post_q.join()
            # 关闭写入队列
            results_q.put(None)
        finally:
            hb.cancel()
            # 等所有消费者退出后再关闭各自的 client
            await asyncio.gather(*grounders, return_exceptions=True)
            await asyncio.gather(*consumers, return_exceptions=True)
            await asyncio.gather(*(c.aclose() for _, c in clients), return_exceptions=True)



def extract_action_fields(text: str):
    # Step 1: extract the JSON inside <answer> ... </answer>
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"action_type":\s*"(.*?)"'
    value_pattern = r'"value":\s*"(.*?)"'
    action_target_pattern = r'"action_target":\s*"(.*?)"'
    answer_match = re.search(answer_tag_pattern, text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        action_type_match = re.search(action_pattern, answer)
        action_type = action_type_match.group(1) if action_type_match else None
        value_match = re.search(value_pattern, answer)
        value = value_match.group(1) if value_match else None
        action_target_match = re.search(action_target_pattern, answer)
        action_target = action_target_match.group(1) if action_target_match else None
        return {
            "action_type": action_type,
            "value": value,
            "action_target": action_target
        }
    else:
        return {
            "action_type": None,
            "value": None,
            "action_target": None
        }



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
        max_tokens=128,
        temperature=0.0,
    )
    return response.choices[0].message.content


async def call_uground(client, expression, image_bs64):
    def pil_to_b64_forug(img: Image.Image) -> str:
        with BytesIO() as image_buffer:
            img.save(image_buffer, format="PNG")
            byte_data = image_buffer.getvalue()
            img_b64 = base64.b64encode(byte_data).decode("utf-8")
        return img_b64

    response = await chat_with_ug(client,
                            image=image_bs64,
                            prompt=expression)

    coordinate = eval(response)
    return coordinate


def main(args):
    MODEL_PATH = args.model_path
    DATA_PATH  = args.data_path
    num_gpus   = args.num_gpus
    start_port = args.start_port


    # 收集所有可用的 chat_url
    base_urls = ['http://127.0.0.1:%d' % (start_port + i) for i in range(num_gpus)]
    chat_urls = [bu + "/v1/chat/completions" for bu in base_urls]
    headers = {"Content-Type": "application/json"}

    ug_client_list = []
    ug_port_start = 8005
    for i in range(num_gpus):
        ug_client = AsyncOpenAI(
            base_url=f'http://127.0.0.1:{ug_port_start+i}/v1',
        )
        ug_client_list.append(ug_client)


    if DATA_PATH.endswith('parquet'):
        data = load_dataset("parquet", data_files=DATA_PATH, split="train")
        data = [dict(r) for r in data]
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    
    if args.end_index > args.start_index:
        data = data[args.start_index:args.end_index]
    print(f"Total samples: {len(data)}")

    OUTPUT_DIR = args.output_path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.end_index > args.start_index:
        NEW_FILE = os.path.join(OUTPUT_DIR, os.path.basename(DATA_PATH).replace(".jsonl", f"_{args.start_index}_{args.end_index}_pred.jsonl").replace('.parquet', f"_{args.start_index}_{args.end_index}_pred.json"))
    else:
        NEW_FILE = os.path.join(OUTPUT_DIR, os.path.basename(DATA_PATH).replace(".jsonl", "_pred.jsonl").replace('.parquet','.json'))

    pbar = tqdm(total=len(data), desc="pipeline(chat.completions)")
    fout = open(NEW_FILE, "w", encoding="utf-8")

    results_q = ThreadQueue(maxsize=8192)
    def writer_thread_fn(fout, pbar, flush_every=100):
        buf, n = [], 0
        while True:
            item = results_q.get()
            if item is None: break
            buf.append(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if n % flush_every == 0:
                fout.writelines(buf); buf.clear()
                fout.flush()
                pbar.update(flush_every)
            results_q.task_done()
        if buf:
            fout.writelines(buf); fout.flush()
            pbar.update(len(buf))

    # 启动线程（在 pipeline_send 之前）
    wt = threading.Thread(target=writer_thread_fn, args=(fout, pbar, 100), daemon=True)
    wt.start()


    try:
        asyncio.run(pipeline_send(
            data_iter=data,                 # 源数据可迭代
            model_path=args.model_path,
            chat_urls=chat_urls,
            ug_client_list=ug_client_list,
            headers=headers,
            cpu_workers=args.cpu_workers,           # e.g., 16/32
            queue_size=args.queue_size,             # e.g., 1024
            async_concurrency=args.async_concurrency,  # e.g., 256
            results_q=results_q,
            max_pixels=args.max_pixels,
        ))
    finally:
        wt.join() 
        fout.close()
        pbar.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--start_port', type=int, default=8001)
    parser.add_argument('--max_pixels', type=int, default=2646000)
    parser.add_argument('--cpu_workers', type=int, default=32)
    parser.add_argument('--async_concurrency', type=int, default=16)
    parser.add_argument('--queue_size', type=int, default=64)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)

    args = parser.parse_args()
    main(args)
