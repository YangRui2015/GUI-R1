# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict, List
import asyncio
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, r1gui_compute_score, r1gui_grounders_compute_score
import threading
import queue

def _run_async(coro):
    """Run an async coroutine from sync code, whether or not a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
        if not loop.is_running():
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

    q: "queue.Queue[object]" = queue.Queue(maxsize=1)
    def worker():
        try:
            q.put(asyncio.run(coro))
        except BaseException as e:
            q.put(e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    res = q.get()
    if isinstance(res, BaseException):
        raise res
    return res

class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str, max_concurrency: int = 8):
        self.tokenizer = tokenizer
        self.computer_score_type = compute_score
        self.max_concurrency = max_concurrency
        if compute_score == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif compute_score == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif compute_score == "r1gui":
            self.compute_score: Callable[[str, str], RewardScore] = r1gui_compute_score
        elif compute_score == "r1gui_grounders":
            self.compute_score: Callable[[str, str, Any], RewardScore] = r1gui_grounders_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics: Dict[str, List[float]] = defaultdict(list)

        # ---- 预处理：长度与解码 ----
        resp_ids_list = data.batch["responses"].tolist()
        resp_mask_list = data.batch["response_mask"].tolist()
        lens = [int(sum(mask)) for mask in resp_mask_list]

        # 批量解码（比单条循环快且简单）
        decoded = self.tokenizer.batch_decode(
            [ids[:L] for ids, L in zip(resp_ids_list, lens)],
            skip_special_tokens=True
        )

        # ---- 计算得分 ----
        if self.computer_score_type == "r1gui_grounders":
            # 并发评分（受信号量限制）
            async def _score_all():
                sem = asyncio.Semaphore(self.max_concurrency)

                async def score_one(i: int):
                    async with sem:
                        pred_str = decoded[i]
                        gt = data[i].non_tensor_batch["ground_truth"]
                        img = data[i].non_tensor_batch.get("image", None)
                        # r1gui_grounders_compute_score 应为 async
                        return await self.compute_score(pred_str, gt, img)

                tasks = [score_one(i) for i in range(B)]
                return await asyncio.gather(*tasks, return_exceptions=False)

            scores = _run_async(_score_all())
            
        else:
            async def _score_sync_all():
                # parallelize sync scoring on a thread pool
                return await asyncio.gather(*[
                    asyncio.to_thread(self.compute_score, decoded[i], data[i].non_tensor_batch["ground_truth"])
                    for i in range(B)
                ])
            scores = _run_async(_score_sync_all())   # <-- no 'await' in sync function

        # ---- 回填 ----
        for i, sc in enumerate(scores):
            L = lens[i]
            if L <= 0:  # 防御：空响应
                continue
            reward_tensor[i, L - 1] = sc["overall"]
            for k, v in sc.items():
                reward_metrics[k].append(v)

        return reward_tensor, reward_metrics
