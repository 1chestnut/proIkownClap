from pprint import pprint
import argparse
import os
import json

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd

from config import conf, common_parameters


def preprocess_dataframe(df, conf):
    """
    Preprocess the AKG dataframe by applying filters and normalization.
    """
    print("Number of rows in the dataset:", len(df))
    print(df.head())

    # 确保有需要的列（如果有多余列，只保留 head, relation, tail）
    needed_cols = ['head', 'relation', 'tail']
    if conf['triple_verification']:
        if 'response' not in df.columns:
            raise ValueError("triple_verification is True but 'response' column not found in data.")
        needed_cols.append('response')
    df = df[needed_cols].copy()

    # Select only verified triples (response == "Yes")
    if conf['triple_verification']:
        df = df[df['response'].str.contains("Yes", na=False)]

    # Filter relations
    df = df[~df['relation'].isin(conf['relations_to_filter'])]
    print("Number of rows after filtering relations:", len(df))

    # Filter rows where tail is "sound" or "Sound"
    df = df[(df['tail'] != 'sound') & (df['tail'] != 'Sound')]

    # --- Post-processing of the triples ---
    # Replace any underscores in the entity and relation names with spaces
    df['head'] = df['head'].str.replace('_', ' ')
    df['tail'] = df['tail'].str.replace('_', ' ')
    df['relation'] = df['relation'].str.replace('_', ' ')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['head', 'relation', 'tail'])
    
    # Reset the index
    df = df.reset_index(drop=True)
    print("Number of rows after post-processing:", len(df))

    # Convert all columns to lowercase
    df['head'] = df['head'].str.lower()
    df['tail'] = df['tail'].str.lower()
    df['relation'] = df['relation'].str.lower()
    
    # Remove any leading or trailing whitespace
    df['head'] = df['head'].str.strip()
    df['tail'] = df['tail'].str.strip()
    df['relation'] = df['relation'].str.strip()
    
    return df


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries in extending parameters for common keys.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = merge_dicts(merged[key], value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Training script for learning KGE models."
    parser.add_argument("--conf_id", required=True,
                        help="Configuration tag in config.py for the experiment")
    args = parser.parse_args()
    return args


def main(conf):
    # ========== 路径处理：将相对路径转换为绝对路径 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    triples_path = conf['triples_path']
    if not os.path.isabs(triples_path):
        triples_path = os.path.join(script_dir, triples_path)
    conf['triples_path'] = triples_path

    test_path = conf['triples_test_path']
    if not os.path.isabs(test_path):
        test_path = os.path.join(script_dir, test_path)
    conf['triples_test_path'] = test_path

    # 结果文件路径
    results_csv_file = conf['results_csv_file']
    if not os.path.isabs(results_csv_file):
        results_csv_file = os.path.join(script_dir, results_csv_file)
    conf['results_csv_file'] = results_csv_file

    # 创建模型输出文件夹
    model_output_folder = os.path.join(
        conf['models_folder'],
        conf['conf_id'],
        f'{conf["job_id"]}_{conf["model_name"]}_{conf["triple_verification"]}')
    os.makedirs(model_output_folder, exist_ok=True)

    # ========== 加载并预处理训练数据 ==========
    # 训练文件无表头，手动指定列名
    df_train = pd.read_csv(conf['triples_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
    df_train = preprocess_dataframe(df_train, conf)

    # 保存处理后的训练数据（用于 PyKEEN）
    train_triples_file = os.path.join(model_output_folder, 'AKG_train_triples.tsv')
    pd.DataFrame({
        'subject': df_train['head'],
        'predicate': df_train['relation'],
        'object': df_train['tail']
    }).to_csv(train_triples_file, sep='\t', header=False, index=False)

    # 创建训练集 TriplesFactory
    training_factory = TriplesFactory.from_path(train_triples_file)

    # ========== 加载并预处理测试数据 ==========
    # 测试文件有表头，读取所有列，然后提取需要的三列
    df_test_raw = pd.read_csv(conf['triples_test_path'], sep='\t', header=0)
    # 选择 head, relation, tail 三列（忽略 label_name, response 等）
    if 'head' not in df_test_raw.columns or 'relation' not in df_test_raw.columns or 'tail' not in df_test_raw.columns:
        raise ValueError("测试文件必须包含 head, relation, tail 列")
    df_test = df_test_raw[['head', 'relation', 'tail']].copy()

    # 对测试数据应用相同的预处理（但关闭 triple_verification，因为测试数据无 response 列）
    conf_test = conf.copy()
    conf_test['triple_verification'] = False
    df_test = preprocess_dataframe(df_test, conf_test)

    # 过滤测试集：只保留实体和关系都在训练集中的三元组
    entity_to_id = training_factory.entity_to_id
    relation_to_id = training_factory.relation_to_id

    def is_in_vocab(row):
        return (row['head'] in entity_to_id) and (row['relation'] in relation_to_id) and (row['tail'] in entity_to_id)

    mask = df_test.apply(is_in_vocab, axis=1)
    filtered_test = df_test[mask].copy()
    filtered_out = df_test[~mask]

    print(f"测试集原始大小: {len(df_test)}")
    print(f"测试集保留大小（实体/关系均在训练集中）: {len(filtered_test)}")
    if len(filtered_test) == 0:
        raise ValueError("所有测试三元组均被过滤，请检查测试集与训练集的实体/关系是否一致。")

    # 保存过滤后的测试数据
    test_triples_file = os.path.join(model_output_folder, 'AKG_test_triples.tsv')
    pd.DataFrame({
        'subject': filtered_test['head'],
        'predicate': filtered_test['relation'],
        'object': filtered_test['tail']
    }).to_csv(test_triples_file, sep='\t', header=False, index=False)

    # 创建测试集 TriplesFactory，使用训练集的映射
    testing_factory = TriplesFactory.from_path(
        test_triples_file,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )

    # ========== 训练模型 ==========
    result = pipeline(
        model=conf['model_name'],
        training=training_factory,
        testing=testing_factory,
        model_kwargs={'embedding_dim': conf['embedding_dim']},
        optimizer_kwargs={'lr': conf['learning_rate']},
        training_kwargs={
            'num_epochs': conf['num_epochs'],
            'batch_size': conf['batch_size'],
        },
        random_seed=42,
        device='cuda'  # 可根据需要改为 'cpu'
    )

    # 保存模型
    result.save_to_directory(model_output_folder)
    print(f"Model saved to {model_output_folder}")

    # ========== 评估 ==========
    evaluator = RankBasedEvaluator()
    metrics = evaluator.evaluate(
        result.model,
        testing_factory.mapped_triples,
        additional_filter_triples=[training_factory.mapped_triples]
    )

    print(f"Hits@1: {metrics.get_metric('hits@1')}")
    print(f"Hits@3: {metrics.get_metric('hits@3')}")
    print(f"Hits@5: {metrics.get_metric('hits@5')}")
    print(f"Hits@10: {metrics.get_metric('hits@10')}")
    print(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")

    # ========== 保存结果到 CSV ==========
    # 确保结果目录存在
    os.makedirs(os.path.dirname(results_csv_file), exist_ok=True)

    try:
        df_results = pd.read_csv(results_csv_file)
    except FileNotFoundError:
        df_results = pd.DataFrame(columns=[
            'job_id', 'conf_id', 'model_name', 'verified',
            'hits@1', 'hits@3', 'hits@5', 'hits@10', 'mean_reciprocal_rank'
        ])

    new_row = {
        'job_id': conf['job_id'],
        'conf_id': conf['conf_id'],
        'model_name': conf['model_name'],
        'verified': conf['triple_verification'],
        'hits@1': metrics.get_metric('hits@1'),
        'hits@3': metrics.get_metric('hits@3'),
        'hits@5': metrics.get_metric('hits@5'),
        'hits@10': metrics.get_metric('hits@10'),
        'mean_reciprocal_rank': metrics.get_metric('mean_reciprocal_rank')
    }

    df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    df_results.to_csv(results_csv_file, index=False)
    print(f"Results saved to {results_csv_file}")


if __name__ == '__main__':
    args = parse_args()
    args = vars(args)
    print('Input arguments:', args)

    conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    conf = {**conf, **args}

    pprint(conf)

    # 保存配置到 JSON
    config_file_path = os.path.join(conf['models_folder'], conf['conf_id'], conf['job_id'])
    os.makedirs(config_file_path, exist_ok=True)
    config_path = os.path.join(config_file_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(conf, f, indent=4)
    print(f"Configuration saved to {config_path}")

    main(conf)