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
import random
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


def process_continuous_feature(x: Any) -> Any:
    """
    处理连续特征（如时间戳、权重），保持原始数值不哈希
    """
    # 处理空值或NaN的情况
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == '':
        return []  # 返回空列表
        
    if isinstance(x, str):
        # 租房数据使用逗号分隔
        x = x.split(',')
    if isinstance(x, list):
        result = []
        for item in x:
            # 处理缺失值
            if str(item).strip() == '-1' or str(item).strip() == '':
                result.append(0)
            else:
                try:
                    # 保持原始数值，不哈希
                    result.append(int(float(item)))
                except (ValueError, TypeError):
                    result.append(0)
        return result
    else:
        if str(x) == '-1' or str(x) == '':
            return 0
        try:
            return int(float(x))
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
        
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        
        original_count = len(self.seq_logs_frame)
        csv_columns = set(self.seq_logs_frame.columns)
        
        # 处理需要embedding的分类特征
        for emb_name, emb_config in embedding_config.items():
            for feature_name in emb_config.feature_names:
                if feature_name in csv_columns:
                    hash_size = emb_config.num_embeddings
                    self.seq_logs_frame[feature_name] = self.seq_logs_frame[feature_name].apply(
                        partial(process_and_hash_x, hash_size=hash_size)
                    )
        
        # 处理连续特征（时间戳、权重、停留时间）- 不需要哈希
        continuous_features = ['timestamp_seq', 'action_weight_seq', 'staytime_seq']
        for feature_name in continuous_features:
            if feature_name in csv_columns:
                # 将字符串转换为数值列表，但不哈希
                self.seq_logs_frame[feature_name] = self.seq_logs_frame[feature_name].apply(
                    process_continuous_feature
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

    def get_item_count(self):
        return len(self.seq_logs_frame)

    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}


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
            # 1. 从历史交互中分离出历史序列数据和候选项
            
            # 房源id特征
            house_history_uih, house_history_candidates = separate_uih_candidates(
                data.houseid_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            
            # 房源属性特征
            cate_history_uih, cate_history_candidates = separate_uih_candidates(
                data.cateid_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            local_history_uih, local_history_candidates = separate_uih_candidates(
                data.localid_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            xq_history_uih, xq_history_candidates = separate_uih_candidates(
                data.xq_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            price_history_uih, price_history_candidates = separate_uih_candidates(
                data.price_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            area_history_uih, area_history_candidates = separate_uih_candidates(
                data.area_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            
            # 时间特征
            timestamps_uih, _ = separate_uih_candidates(
                data.timestamp_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            
            # 权重和停留时间
            action_weights_uih, action_weights_candidates = separate_uih_candidates(
                data.action_weight_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            staytime_uih, staytime_candidates = separate_uih_candidates(
                data.staytime_seq,
                candidates_max_seq_len=max_num_candidates,
            )
            

            # 2. 截断所有序列到最大UIH长度
            house_history_uih = maybe_truncate_seq(house_history_uih, self._max_uih_len)
            cate_history_uih = maybe_truncate_seq(cate_history_uih, self._max_uih_len)
            local_history_uih = maybe_truncate_seq(local_history_uih, self._max_uih_len)
            xq_history_uih = maybe_truncate_seq(xq_history_uih, self._max_uih_len)
            price_history_uih = maybe_truncate_seq(price_history_uih, self._max_uih_len)
            area_history_uih = maybe_truncate_seq(area_history_uih, self._max_uih_len)
            timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
            action_weights_uih = maybe_truncate_seq(action_weights_uih, self._max_uih_len)
            staytime_uih = maybe_truncate_seq(staytime_uih, self._max_uih_len)
            # dummy_weights_uih = maybe_truncate_seq(dummy_weights_uih, self._max_uih_len)
            # dummy_staytime_uih = maybe_truncate_seq(dummy_staytime_uih, self._max_uih_len)

            uih_seq_len = len(house_history_uih)
            
            # 3. 验证所有序列长度一致性
            assert uih_seq_len == len(cate_history_uih), "cate history len differs from house len."
            assert uih_seq_len == len(local_history_uih), "local history len differs from house len."
            assert uih_seq_len == len(xq_history_uih), "xq history len differs from house len."
            assert uih_seq_len == len(price_history_uih), "price history len differs from house len."
            assert uih_seq_len == len(area_history_uih), "area history len differs from house len."
            assert uih_seq_len == len(timestamps_uih), "history len differs from timestamp len."
            assert uih_seq_len == len(action_weights_uih), "history len differs from real weights len."
            assert uih_seq_len == len(staytime_uih), "history len differs from real staytime len."

            # 4. 构建UIH特征的KeyedJaggedTensor
            uih_kjt_values: List = []
            uih_kjt_lengths: List = []
            
            # 添加上下文特征（如用户ID等标量特征）
            for name, length in self._contextual_feature_to_max_length.items():
                uih_kjt_values.append(data[name])
                uih_kjt_lengths.append(length)

            # 添加序列特征
            uih_kjt_values.extend(
                house_history_uih + cate_history_uih + local_history_uih + xq_history_uih +
                price_history_uih + area_history_uih + timestamps_uih 
                + action_weights_uih + staytime_uih  # 历史序列数据使用真实权重和停留时间
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

            # 5. 构建候选特征的KeyedJaggedTensor（包含虚拟特征和真实特征）
            candidates_kjt_lengths = max_num_candidates * torch.ones(
                len(self._candidates_keys), dtype=torch.long
            )
            candidates_kjt_values = (
                house_history_candidates +
                cate_history_candidates +
                local_history_candidates +
                xq_history_candidates +
                price_history_candidates +
                area_history_candidates +
                [dummy_query_time] * max_num_candidates +
                [0] * max_num_candidates +  # 使用常数0填充虚拟权重
                [0] * max_num_candidates + # 使用常数0填充虚拟停留时间
                # dummy_weights_candidates +          # 虚拟权重（用于embedding）
                # dummy_staytime_candidates            # 虚拟停留时间（用于embedding）
                action_weights_candidates +    # 真实权重（仅用于标签生成）
                staytime_candidates            # 真实停留时间（仅用于标签生成）
            )
            candidates_features_kjt = KeyedJaggedTensor(
                keys=self._candidates_keys,
                lengths=candidates_kjt_lengths,
                values=torch.tensor(candidates_kjt_values, dtype=torch.long),
            )

        return uih_features_kjt, candidates_features_kjt
