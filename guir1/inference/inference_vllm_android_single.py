import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
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

# ====== Stage1: CPU 预处理函数（与之前相同，尽量只做 bytes->JPEG->b64）======
def _preprocess_to_b64(processor, sample):
    img = Image.open(BytesIO(sample["image"]["bytes"]))
    w, h = img.size
    # resize image if too large using processor
    msg = [{"role":"user","content":[
        {"type":"image", "image": img},
        {"type":"text", "text": "."},
    ]}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(msg, return_video_kwargs=True)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors="pt")
    grid_h, grid_w = inputs["image_grid_thw"][0][1].item(), inputs["image_grid_thw"][0][2].item()
    patch = int(getattr(processor.image_processor, "patch_size", 14))
    new_h, new_w = grid_h * patch, grid_w * patch
    if h > new_h or w > new_w:
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    # convert to base64
    buf = BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    out = dict(sample)
    out["image_b64"] = b64
    out["orig_w"] = w
    out["orig_h"] = h
    out["image"] = ""  # 或 del out["image"]
    out["scale"]=[new_w/w, new_h/h]
    out["image_size"]=[new_w, new_h]
    print(f"Preprocessed image: {w}x{h} -> {new_w}x{new_h}, processor max pixels={processor.image_processor.max_pixels}")
    return out



def _build_payload(sample, model_path):
    instr = sample["instruction"]
    history = sample.get("history", "None")
    user_text = (
        "You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot, "
        f"I want you to continue executing the command '{instr}', with the action history being '{history}'.\n"
        "Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll']), "
        "the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
        "Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:\n"
        "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
        "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
        "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )
    return {
        "model": model_path,
        "messages": [
            {"role":"user","content":[
                {"type":"text","text": user_text},
                {"type":"image_url","image_url":{"url": f"data:image/png;base64,{sample['image_b64']}"}}
            ]}
        ],
        "temperature": 0.0,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "max_tokens": 1024,
    }

# ====== Stage2: 异步发送（消费者）======
async def _send_one(client, url, headers, payload, meta, parse_fn):
    print(f"Sending to {url} ...")
    r = await client.post(url, headers=headers, content=json.dumps(payload),
                            timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0),)
    if r.status_code != 200:
        # 打印 server 提示，直指问题
        print(f"[{url}] {r.status_code} {r.reason_phrase} -> {r.text[:500]}")
        r.raise_for_status()

    data_json = r.json()
    text = data_json["choices"][0]["message"]["content"]
    # 解析动作/坐标/文本（复用你现有的函数）
    pred_coord, _ok = parse_fn["coord"](text)
    x, y = pred_coord[0], pred_coord[1]
    out = dict(meta)
    out["pred"] = text
    x = int(x / meta["scale"][0])
    y = int(y / meta["scale"][1])
    out["pred_coord"] = [x, y]
    out["pred_action"] = parse_fn["action"](text)
    out["pred_input_text"] = parse_fn["input_text"](text)
    return out

# ====== 流水线：生产者-消费者 ======
async def pipeline_send(data_iter, model_path, chat_urls, headers,
                        cpu_workers=16, queue_size=2048,
                        async_concurrency=256, 
                        parse_fn=None, results_q=None, max_pixels=2646000):
    """
    data_iter: 可迭代的原始样本
    cpu_workers: Stage1 进程数
    queue_size: 队列容量（防止内存暴涨，提供背压）
    async_concurrency: Stage2 并发度
    """

    limits = httpx.Limits(max_keepalive_connections=512, max_connections=512)
    q = asyncio.Queue(maxsize=queue_size)
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
            print(f"[HB] qsize={q.qsize()}/{q.maxsize}  inflight_total={inflight_total}  {by_url_str}  consumers={num_consumers}")

    # 1) 生产者：多进程预处理 -> 放入队列
    def produce_batched(executor, it, batch=64):
        processor = AutoProcessor.from_pretrained(model_path)
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
            try:
                await inc(url)          
                res = await _send_one(client, url, headers, payload, meta, parse_fn)
                results_q.put(res)
            except Exception as e:
                print(f"[CONS#{cid}@{url}] send error:", repr(e))
            finally:
                await dec(url)   
                q.task_done()

    # 3) 并发运行
    default_timeout = httpx.Timeout(connect=20.0, read=300.0, write=300.0, pool=20.0)
    limits = httpx.Limits(max_keepalive_connections=128, max_connections=128)

    with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
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

        async def producer_loop(executor):
            produced = 0
            for fut_batch in produce_batched(executor, data_iter, batch=32):  # 可把 64 -> 128
                for fut in as_completed(fut_batch):
                    try:
                        s = fut.result()
                    except Exception as e:
                        print("[PROD] preprocess error:", repr(e))
                        continue
                    payload = _build_payload(s, model_path)
                    s.pop("image_b64", None)
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
            # 关闭写入队列
            results_q.put(None)
        finally:
            hb.cancel()
            # 等所有消费者退出后再关闭各自的 client
            await asyncio.gather(*consumers, return_exceptions=True)
            await asyncio.gather(*(c.aclose() for _, c in clients), return_exceptions=True)



def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return None

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"


def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False



def main(args):
    MODEL_PATH = args.model_path
    DATA_PATH  = args.data_path

    num_gpus   = args.num_gpus
    start_port = args.start_port


    # 收集所有可用的 chat_url
    base_urls = ['http://127.0.0.1:%d' % (start_port + i) for i in range(num_gpus)]
    chat_urls = [bu + "/v1/chat/completions" for bu in base_urls]
    headers = {"Content-Type": "application/json"}


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
            headers=headers,
            cpu_workers=args.cpu_workers,           # e.g., 16/32
            queue_size=args.queue_size,             # e.g., 1024
            async_concurrency=args.async_concurrency,  # e.g., 256
            parse_fn=parse_fn,
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
    parser.add_argument('--cpu_workers', type=int, default=16)
    parser.add_argument('--async_concurrency', type=int, default=4)
    parser.add_argument('--queue_size', type=int, default=128)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)

    args = parser.parse_args()
    main(args)
