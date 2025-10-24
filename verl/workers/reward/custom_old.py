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
from typing import Any, Callable, Dict, Tuple, TypedDict
import asyncio
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, r1gui_compute_score, r1gui_grounders_compute_score


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


async def compute_scores_batch_async(
    batch_items: List[Dict[str, Any]],
    computer_score_type: str = "r1gui_grounders",
    max_concurrency: int = MAX_CONCURRENCY,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    batch_items: 每个元素形如：
      {
        "response_ids": List[int],
        "response_mask": List[int] (0/1),
        "ground_truth": str(JSON),
        "image": {"bytes": b"..."}  # 仅在 r1gui_grounders 用到
      }
    """
    ug_clients = build_clients(UG_URLS)
    router = UgRouter(ug_clients)

    # 1) 计算有效长度
    valid_lengths = []
    for it in batch_items:
        L = int(sum(it["response_mask"]))
        valid_lengths.append(L)

    # 2) 并发解码（to_thread）
    async def decode_one(ids, L):
        return await asyncio.to_thread(tokenizer.decode, ids[:L], True)

    decoded = await asyncio.gather(*[
        decode_one(it["response_ids"], valid_lengths[i])
        for i, it in enumerate(batch_items)
    ])

    # 3) 并发评分（限流）
    sem = asyncio.Semaphore(max_concurrency)

    async def score_one(i):
        async with sem:
            pred_str = decoded[i]
            gt = batch_items[i]["ground_truth"]
            if computer_score_type == "r1gui_grounders":
                img = batch_items[i].get("image", None)
                return await r1gui_grounders_compute_score(pred_str, gt, img, router)
            else:
                # 其它评分：若是同步函数也可用 to_thread 包一下
                return await asyncio.to_thread(lambda: {"overall": 0.0, "format": 0.0, "accuracy": 0.0})

    scores = await asyncio.gather(*[score_one(i) for i in range(len(batch_items))])

    # 4) 关闭http客户端
    for c in ug_clients:
        try:
            await c.http_client.aclose()
        except Exception:
            pass

    return scores, valid_lengths


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str):
        self.tokenizer = tokenizer
        self.computer_score_type = compute_score
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
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            
            if self.computer_score_type == "r1gui_grounders":
                image_byte = data_item.non_tensor_batch.get("image", None)
                score = asyncio.run(self.compute_score(response_str, ground_truth, image_byte))
            else:
                score = self.compute_score(response_str, ground_truth)

            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
