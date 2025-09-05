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

# pyre-strict
from typing import Dict

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

HSTU_EMBEDDING_DIM = 256
HASH_SIZE = 10_000_000


def get_hstu_configs(dataset: str = "debug") -> DlrmHSTUConfig:
    hstu_config = DlrmHSTUConfig(
        hstu_num_heads=4,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=128,
        hstu_attn_num_layers=3,
        hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
        hstu_preprocessor_hidden_dim=256,
        hstu_transducer_embedding_dim=512,
        hstu_group_norm=True,
        hstu_input_dropout_ratio=0.2,
        hstu_linear_dropout_rate=0.1,
        causal_multitask_weights=0.2,
    )
    if "movielens" in dataset:
        assert dataset in ["movielens-1m", "movielens-20m", "movielens-13b"]
        hstu_config.user_embedding_feature_names = (
            [
                "movie_id",
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
            ]
            if dataset == "movielens-1m"
            else [
                "movie_id",
                "user_id",
            ]
        )
        hstu_config.item_embedding_feature_names = [
            "item_movie_id",
        ]
        hstu_config.uih_post_id_feature_name = "movie_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_dummy_weights"
        hstu_config.candidates_watchtime_feature_name = "item_dummy_watchtime"
        hstu_config.contextual_feature_to_max_length = (
            {
                "user_id": 1,
                "sex": 1,
                "age_group": 1,
                "occupation": 1,
                "zip_code": 1,
            }
            if dataset == "movielens-1m"
            else {
                "user_id": 1,
            }
        )
        hstu_config.contextual_feature_to_min_uih_length = (
            {
                "user_id": 20,
                "sex": 20,
                "age_group": 20,
                "occupation": 20,
                "zip_code": 20,
            }
            if dataset == "movielens-1m"
            else {
                "user_id": 20,
            }
        )
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("movie_id", "item_movie_id"),
            ("action_timestamp", "item_query_time"),
            ("dummy_weights", "item_dummy_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
        ]
        hstu_config.hstu_uih_feature_names = (
            [
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "dummy_weights",
                "dummy_watch_time",
            ]
            if dataset == "movielens-1m"
            else [
                "user_id",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "dummy_weights",
                "dummy_watch_time",
            ]
        )
        hstu_config.hstu_candidate_feature_names = [
            "item_movie_id",
            "item_query_time",
            "item_dummy_weights",
            "item_dummy_watchtime",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="rating",
                task_weight=1,
                task_type=MultitaskTaskType.REGRESSION,
            )
        ]
    elif "kuairand" in dataset:
        hstu_config.user_embedding_feature_names = [
            "video_id",
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_video_id",
        ]
        hstu_config.uih_post_id_feature_name = "video_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.uih_weight_feature_name = "action_weight"
        hstu_config.candidates_weight_feature_name = "item_action_weight"
        hstu_config.candidates_watchtime_feature_name = "item_target_watchtime"
        # There are more contextual features in the dataset, see https://kuairand.com/ for details
        hstu_config.contextual_feature_to_max_length = {
            "user_id": 1,
            "user_active_degree": 1,
            "follow_user_num_range": 1,
            "fans_user_num_range": 1,
            "friend_user_num_range": 1,
            "register_days_range": 1,
        }
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("video_id", "item_video_id"),
            ("action_timestamp", "item_query_time"),
            ("action_weight", "item_action_weight"),
            ("watch_time", "item_target_watchtime"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "video_id",
            "action_timestamp",
            "action_weight",
            "watch_time",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_video_id",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="is_click",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_like",
                task_weight=2,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_follow",
                task_weight=4,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_comment",
                task_weight=8,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_forward",
                task_weight=16,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_hate",
                task_weight=32,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="long_view",
                task_weight=64,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_profile_enter",
                task_weight=128,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
        ]
        hstu_config.action_weights = [1, 2, 4, 8, 16, 32, 64, 128]
    elif dataset == "zufang_simple":
        # 租房数据集专用配置
        hstu_config.user_embedding_feature_names = [
            "houseid_seq",  # 对应video_id
            "user_id",          # 用户ID - 唯一的用户静态特征
        ]
        hstu_config.item_embedding_feature_names = [
            "item_house_id",    # 候选房源ID
            "item_cate_id",     # 候选房源类别
            "item_local_id",    # 候选房源位置
            "item_xq_id",       # 候选房源小区
            "item_price",       # 候选房源价格
            "item_area",        # 候选房源面积
        ]
        hstu_config.uih_post_id_feature_name = "houseid_seq"
        hstu_config.uih_action_time_feature_name = "timestamp_seq"
        hstu_config.uih_weight_feature_name = "action_weight_seq"
        
        # 标签生成使用真实特征，模型输入使用虚拟特征
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_real_weights"     # 真实权重用于标签生成
        hstu_config.candidates_watchtime_feature_name = "item_real_staytime"  # 真实停留时间用于标签生成

        # 上下文特征配置 - 只有用户静态特征
        hstu_config.contextual_feature_to_max_length = {
            "user_id": 1,       # 用户ID在每个样本中只出现1次
        }
        hstu_config.contextual_feature_to_min_uih_length = {
            "user_id": 15,      
        }
        
        # 特征映射关系 - 注意：真实权重和停留时间不参与embedding
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("houseid_seq", "item_house_id"),
            ("cateid_seq", "item_cate_id"),
            ("localid_seq", "item_local_id"),
            ("xq_seq", "item_xq_id"),
            ("price_seq", "item_price"),
            ("area_seq", "item_area"),
            ("timestamp_seq", "item_query_time"),
            ("action_weight_seq", "item_dummy_weights"),     # 虚拟权重用于模型输入
            ("staytime_seq", "item_dummy_staytime"),   # 虚拟停留时间用于模型输入
        ]
        
        # 用户交互历史特征 
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "houseid_seq",
            "cateid_seq", 
            "localid_seq",
            "xq_seq",
            "price_seq",
            "area_seq",
            "timestamp_seq",
            "action_weight_seq",     
            "staytime_seq",     
        ]
        
        # 候选物品特征 - 虚拟特征用于embedding，真实特征用于标签生成
        hstu_config.hstu_candidate_feature_names = [
            "item_house_id",
            "item_cate_id",
            "item_local_id", 
            "item_xq_id",
            "item_price",
            "item_area",
            "item_query_time",
            "item_dummy_weights",    # 虚拟权重（用于embedding）
            "item_dummy_staytime",   # 虚拟停留时间（用于embedding）
            "item_real_weights",     # 真实权重（用于标签生成）
            "item_real_staytime",    # 真实停留时间（用于标签生成）
        ]
        
        # 多任务配置 - 保留原有任务
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="is_click",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_like", 
                task_weight=2,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
        ]
        
        # 权重配置 - 对应任务权重
        hstu_config.action_weights = [1, 2]
    else:
        hstu_config.user_embedding_feature_names = [
            "uih_post_id",
            "uih_owner_id",
            "viewer_id",
            "dummy_contexual",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_post_id",
            "item_owner_id",
        ]
        hstu_config.uih_post_id_feature_name = "uih_post_id"
        hstu_config.uih_action_time_feature_name = "uih_action_time"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weight"
        hstu_config.candidates_watchtime_feature_name = "item_target_watchtime"
        hstu_config.contextual_feature_to_max_length = {
            "viewer_id": 1,
            "dummy_contexual": 1,
        }
        hstu_config.contextual_feature_to_min_uih_length = {
            "viewer_id": 128,
            "dummy_contexual": 128,
        }
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("uih_post_id", "item_post_id"),
            ("uih_owner_id", "item_owner_id"),
            ("uih_action_time", "item_query_time"),
            ("uih_weight", "item_action_weight"),
            ("uih_watchtime", "item_target_watchtime"),
            ("uih_video_length", "item_video_length"),
            ("uih_surface_type", "item_surface_type"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "uih_post_id",
            "uih_action_time",
            "uih_weight",
            "uih_owner_id",
            "uih_watchtime",
            "uih_surface_type",
            "uih_video_length",
            "viewer_id",
            "dummy_contexual",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_post_id",
            "item_owner_id",
            "item_surface_type",
            "item_video_length",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="vvp100",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
    return hstu_config


def get_embedding_table_config(dataset: str = "debug") -> Dict[str, EmbeddingConfig]:
    if "movielens" in dataset:
        assert dataset in ["movielens-1m", "movielens-20m", "movielens-13b"]
        return (
            {
                "movie_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="movie_id",
                    data_type=DataType.FP16,
                    feature_names=["movie_id", "item_movie_id"],
                ),
                "user_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="user_id",
                    data_type=DataType.FP16,
                    feature_names=["user_id"],
                ),
                "sex": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="sex",
                    data_type=DataType.FP16,
                    feature_names=["sex"],
                ),
                "age_group": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="age_group",
                    data_type=DataType.FP16,
                    feature_names=["age_group"],
                ),
                "occupation": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="occupation",
                    data_type=DataType.FP16,
                    feature_names=["occupation"],
                ),
                "zip_code": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="zip_code",
                    data_type=DataType.FP16,
                    feature_names=["zip_code"],
                ),
            }
            if dataset == "movielens-1m"
            else {
                "movie_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="movie_id",
                    data_type=DataType.FP16,
                    feature_names=["movie_id", "item_movie_id"],
                ),
                "user_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=HSTU_EMBEDDING_DIM,
                    name="user_id",
                    data_type=DataType.FP16,
                    feature_names=["user_id"],
                ),
            }
        )
    elif "kuairand" in dataset:
        return {
            "video_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="video_id",
                data_type=DataType.FP16,
                feature_names=["video_id", "item_video_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
            "user_active_degree": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_active_degree",
                data_type=DataType.FP16,
                feature_names=["user_active_degree"],
            ),
            "follow_user_num_range": EmbeddingConfig(
                num_embeddings=9,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="follow_user_num_range",
                data_type=DataType.FP16,
                feature_names=["follow_user_num_range"],
            ),
            "fans_user_num_range": EmbeddingConfig(
                num_embeddings=9,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="fans_user_num_range",
                data_type=DataType.FP16,
                feature_names=["fans_user_num_range"],
            ),
            "friend_user_num_range": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="friend_user_num_range",
                data_type=DataType.FP16,
                feature_names=["friend_user_num_range"],
            ),
            "register_days_range": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="register_days_range",
                data_type=DataType.FP16,
                feature_names=["register_days_range"],
            ),
        }
    elif dataset == "zufang_simple":
        # 租房数据集的embedding配置 - 包含虚拟特征
        return {
            "houseid_seq": EmbeddingConfig(
                num_embeddings=HASH_SIZE,  # 房源ID数量很大，使用哈希
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="houseid_seq",
                data_type=DataType.FP16,
                feature_names=["houseid_seq", "item_house_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,  # 用户ID数量很大，使用哈希
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id", 
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
            "cateid_seq": EmbeddingConfig(
                num_embeddings=20,  # 类别数量很少，留些余量
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="cateid_seq",
                data_type=DataType.FP16,
                feature_names=["cateid_seq", "item_cate_id"],
            ),
            "localid_seq": EmbeddingConfig(
                num_embeddings=50000,  # 位置数量中等
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="localid_seq",
                data_type=DataType.FP16,
                feature_names=["localid_seq", "item_local_id"],
            ),
            "xq_seq": EmbeddingConfig(
                num_embeddings=HASH_SIZE,  # 小区数量较大，使用哈希
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="xq_seq",
                data_type=DataType.FP16,
                feature_names=["xq_seq", "item_xq_id"],
            ),
            "price_seq": EmbeddingConfig(
                num_embeddings=15,  # 价格区间数量适中
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="price_seq",
                data_type=DataType.FP16,
                feature_names=["price_seq", "item_price"],
            ),
            "area_seq": EmbeddingConfig(
                num_embeddings=10,  # 面积区间数量适中
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="area_seq",
                data_type=DataType.FP16,
                feature_names=["area_seq", "item_area"],
            ),
        }
    else:
        return {
            "post_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="post_id",
                data_type=DataType.FP16,
                feature_names=[
                    "uih_post_id",
                    "item_post_id",
                    "uih_owner_id",
                    "item_owner_id",
                ],
            ),
            "viewer_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="viewer_id",
                data_type=DataType.FP16,
                feature_names=["viewer_id"],
            ),
            "dummy_contexual": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="dummy_contexual",
                data_type=DataType.FP16,
                feature_names=["dummy_contexual"],
            ),
        }
