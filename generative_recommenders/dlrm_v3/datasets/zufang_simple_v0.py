# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
import json
import time
from functools import partial
from typing import Any, Dict, List

import pandas as pd
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.utils import (
    maybe_truncate_seq,
    separate_uih_candidates,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def process_and_hash_x(x: Any, hash_size: int) -> Any:
    """
    处理租房数据的哈希函数，支持逗号分隔的字符串格式
    与KuaiRand不同，租房数据使用逗号分隔而不是JSON格式
    """
    # 处理空值或NaN的情况
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == '':
        return []  # 返回空列表
        
    if isinstance(x, str):
        # 租房数据使用逗号分隔，不是JSON格式
        x = x.split(',')
    if isinstance(x, list):
        result = []
        for item in x:
            # 处理 -1 缺失值
            if str(item).strip() == '-1' or str(item).strip() == '':
                result.append(0)
            else:
                try:
                    result.append(int(float(item)) % hash_size)
                except (ValueError, TypeError):
                    result.append(0)
        return result  # 关键：返回列表，不是单个值
    else:
        if str(x) == '-1' or str(x) == '':
            return 0
        try:
            return int(float(x)) % hash_size
        except (ValueError, TypeError):
            return 0


class DLRMv3ZufangSimpleDataset(DLRMv3RandomDataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        embedding_config: Dict[str, Any],
        seq_logs_file: str,
        is_inference: bool,
        **kwargs,
    ) -> None:
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self.seq_logs_frame: pd.DataFrame = pd.read_csv(seq_logs_file, delimiter=",")
        
        # 数据预处理：移除无效数据，防止dummy样本进入训练
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        
        original_count = len(self.seq_logs_frame)
        
        # 先应用哈希处理，获得处理后的序列
        for key, table in embedding_config.items():
            if key in self.seq_logs_frame.columns:
                hash_size = table.num_embeddings
                self.seq_logs_frame[key] = self.seq_logs_frame[key].apply(
                    partial(process_and_hash_x, hash_size=hash_size)
                )
        
        # 过滤掉会被跳过的样本：空序列或短序列
        def is_valid_sample(row):
            houseid_seq = row.houseid_seq
            return (isinstance(houseid_seq, list) and 
                    len(houseid_seq) > max_num_candidates)
        
        valid_mask = self.seq_logs_frame.apply(is_valid_sample, axis=1)
        self.seq_logs_frame = self.seq_logs_frame[valid_mask].reset_index(drop=True)
        
        filtered_count = len(self.seq_logs_frame)
        print(f"数据预处理完成：原始 {original_count} 条 → 有效 {filtered_count} 条 "
              f"(过滤了 {original_count - filtered_count} 条无效样本)")
        
        # 移除哈希处理代码，因为已经在上面处理了
        
        print(f"数据预处理完成：原始 {original_count} 条 → 有效 {filtered_count} 条 "
              f"(过滤了 {original_count - filtered_count} 条无效样本)")

    def get_item_count(self):
        return len(self.seq_logs_frame)

    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    # def load_query_samples(self, sample_list):
    #     max_num_candidates = (
    #         self._max_num_candidates_inference
    #         if self._is_inference
    #         else self._max_num_candidates
    #     )
    #     self.items_in_memory = {}
    #     for idx in sample_list:
    #         data = self.seq_logs_frame.iloc[idx]
    #         # 检查houseid_seq序列长度（对应KuaiRand的video_id检查）
    #         houseid_seq = data.houseid_seq
            
    #         # 处理空序列的情况：跳过空值、NaN或空列表
    #         if (houseid_seq is None or 
    #             (isinstance(houseid_seq, float) and pd.isna(houseid_seq)) or
    #             not isinstance(houseid_seq, list) or 
    #             len(houseid_seq) == 0 or
    #             len(houseid_seq) <= max_num_candidates):
    #             continue  # 直接跳过，与KuaiRand一致
                
    #         sample = self.load_item(data, max_num_candidates)
    #         self.items_in_memory[idx] = sample

    #     self.last_loaded = time.time()

    def load_query_samples(self, sample_list):
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for idx in sample_list:
            data = self.seq_logs_frame.iloc[idx]
            # 由于预处理已经过滤了无效样本，这里直接加载
            sample = self.load_item(data, max_num_candidates)
            self.items_in_memory[idx] = sample

        self.last_loaded = time.time()

    def load_item(self, data, max_num_candidates):
        with torch.profiler.record_function("load_item_zufang"):
            # 1. 分离核心序列特征（对应KuaiRand的处理）
            # 确保所有序列特征都不为空，为空则用空列表替代
            # houseid_seq = data.houseid_seq if isinstance(data.houseid_seq, list) else []
            # action_weight_seq = data.action_weight_seq if isinstance(data.action_weight_seq, list) else []
            # timestamp_seq = data.timestamp_seq if isinstance(data.timestamp_seq, list) else []
            # staytime_seq = data.staytime_seq if isinstance(data.staytime_seq, list) else []
            
            # houseid_seq -> video_id
            house_history_uih, house_history_candidates = separate_uih_candidates(
                data.houseid_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            # action_weight_seq -> action_weights
            action_weights_uih, action_weights_candidates = separate_uih_candidates(
                data.action_weight_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            # timestamp_seq -> time_ms
            timestamps_uih, _ = separate_uih_candidates(
                data.timestamp_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            # staytime_seq -> play_time_ms
            staytime_uih, staytime_candidates = separate_uih_candidates(
                data.staytime_seq,
                candidates_max_seq_len=max_num_candidates,
            )

            # 2. 截断序列到最大UIH长度
            house_history_uih = maybe_truncate_seq(house_history_uih, self._max_uih_len)
            action_weights_uih = maybe_truncate_seq(action_weights_uih, self._max_uih_len)
            timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
            staytime_uih = maybe_truncate_seq(staytime_uih, self._max_uih_len)

            uih_seq_len = len(house_history_uih)
            
            # 3. 验证序列长度一致性
            assert uih_seq_len == len(timestamps_uih), "history len differs from timestamp len."
            assert uih_seq_len == len(action_weights_uih), "history len differs from weights len."
            assert uih_seq_len == len(staytime_uih), "history len differs from staytime len."

            # 4. 构建UIH特征的KeyedJaggedTensor
            uih_kjt_values: List = []
            uih_kjt_lengths: List = []
            
            # 添加上下文特征（如用户ID等标量特征）
            for name, length in self._contextual_feature_to_max_length.items():
                uih_kjt_values.append(data[name])
                uih_kjt_lengths.append(length)

            # 添加序列特征（与KuaiRand相同的顺序）
            uih_kjt_values.extend(
                house_history_uih + timestamps_uih + action_weights_uih + staytime_uih
            )

            # 计算序列特征的长度
            uih_kjt_lengths.extend([
                uih_seq_len for _ in range(
                    len(self._uih_keys) - len(self._contextual_feature_to_max_length)
                )
            ])

            dummy_query_time = max(timestamps_uih) if timestamps_uih else 0
            uih_features_kjt = KeyedJaggedTensor(
                keys=self._uih_keys,
                lengths=torch.tensor(uih_kjt_lengths, dtype=torch.long),
                values=torch.tensor(uih_kjt_values, dtype=torch.long),
            )

            # 5. 构建候选特征的KeyedJaggedTensor
            candidates_kjt_lengths = max_num_candidates * torch.ones(
                len(self._candidates_keys), dtype=torch.long
            )
            candidates_kjt_values = (
                house_history_candidates
                + action_weights_candidates
                + staytime_candidates
                + [dummy_query_time] * max_num_candidates
            )
            candidates_features_kjt = KeyedJaggedTensor(
                keys=self._candidates_keys,
                lengths=candidates_kjt_lengths,
                values=torch.tensor(candidates_kjt_values, dtype=torch.long),
            )

        return uih_features_kjt, candidates_features_kjt
