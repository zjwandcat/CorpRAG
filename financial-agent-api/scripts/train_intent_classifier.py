"""
意图分类模型训练脚本

使用TF-IDF向量化 + 朴素贝叶斯分类器训练意图识别模型。
支持6种意图分类:报销咨询、请假流程、IT支持、人事管理、财务管理、其他

Usage:
    python train_intent_classifier.py --output-path ./intent_model.pkl --verbose
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def generate_training_data() -> List[Tuple[str, str]]:
    """
    生成合成训练数据
    
    Returns:
        训练数据列表,每个元素为(文本, 标签)元组
    """
    training_data = [
        # 报销咨询 (20条)
        ("如何报销差旅费?", "报销咨询"),
        ("报销流程是什么?", "报销咨询"),
        ("差旅费怎么报销?", "报销咨询"),
        ("发票抬头是什么?", "报销咨询"),
        ("报销需要哪些材料?", "报销咨询"),
        ("餐饮费能报销吗?", "报销咨询"),
        ("交通费报销标准是多少?", "报销咨询"),
        ("报销审批需要多长时间?", "报销咨询"),
        ("住宿费报销限额是多少?", "报销咨询"),
        ("如何提交报销申请?", "报销咨询"),
        ("电子发票可以报销吗?", "报销咨询"),
        ("报销单在哪里下载?", "报销咨询"),
        ("培训费用如何报销?", "报销咨询"),
        ("加班餐费能报销吗?", "报销咨询"),
        ("出差补贴怎么算?", "报销咨询"),
        ("报销被驳回了怎么办?", "报销咨询"),
        ("医疗费用能报销吗?", "报销咨询"),
        ("通讯费报销标准是多少?", "报销咨询"),
        ("办公用品如何报销?", "报销咨询"),
        ("招待费报销流程是什么?", "报销咨询"),
        
        # 请假流程 (20条)
        ("请假需要提前几天申请?", "请假流程"),
        ("年假怎么休?", "请假流程"),
        ("病假需要提供证明吗?", "请假流程"),
        ("事假扣工资吗?", "请假流程"),
        ("婚假有多少天?", "请假流程"),
        ("产假怎么申请?", "请假流程"),
        ("陪产假有几天?", "请假流程"),
        ("丧假怎么申请?", "请假流程"),
        ("调休怎么操作?", "请假流程"),
        ("请假审批流程是什么?", "请假流程"),
        ("年假余额怎么查询?", "请假流程"),
        ("病假工资怎么算?", "请假流程"),
        ("事假最多能请几天?", "请假流程"),
        ("请假单在哪里填写?", "请假流程"),
        ("哺乳假怎么申请?", "请假流程"),
        ("工伤假怎么处理?", "请假流程"),
        ("请假被拒绝了怎么办?", "请假流程"),
        ("可以撤销请假申请吗?", "请假流程"),
        ("节假日加班怎么调休?", "请假流程"),
        ("请假需要谁审批?", "请假流程"),
        
        # IT支持 (20条)
        ("电脑无法连接VPN", "IT支持"),
        ("忘记密码怎么办?", "IT支持"),
        ("打印机无法使用", "IT支持"),
        ("网络连接不上", "IT支持"),
        ("邮箱无法登录", "IT支持"),
        ("软件安装失败", "IT支持"),
        ("系统运行缓慢", "IT支持"),
        ("U盘无法识别", "IT支持"),
        ("显示器不亮", "IT支持"),
        ("键盘失灵了", "IT支持"),
        ("鼠标没反应", "IT支持"),
        ("蓝屏了怎么办", "IT支持"),
        ("无法访问内网", "IT支持"),
        ("VPN连接经常断开", "IT支持"),
        ("Office激活失败", "IT支持"),
        ("杀毒软件报警", "IT支持"),
        ("数据备份怎么做?", "IT支持"),
        ("远程桌面怎么用?", "IT支持"),
        ("WiFi密码是多少?", "IT支持"),
        ("新员工账号怎么开通?", "IT支持"),
        
        # 人事管理 (20条)
        ("入职手续怎么办?", "人事管理"),
        ("离职流程是什么?", "人事管理"),
        ("转正申请怎么提交?", "人事管理"),
        ("劳动合同在哪里签?", "人事管理"),
        ("社保怎么缴纳?", "人事管理"),
        ("公积金怎么查询?", "人事管理"),
        ("工资条在哪里看?", "人事管理"),
        ("考勤记录怎么查询?", "人事管理"),
        ("加班申请怎么提交?", "人事管理"),
        ("绩效评估流程是什么?", "人事管理"),
        ("晋升条件有哪些?", "人事管理"),
        ("调岗怎么申请?", "人事管理"),
        ("员工档案在哪里?", "人事管理"),
        ("入职体检要求是什么?", "人事管理"),
        ("离职证明怎么开?", "人事管理"),
        ("档案转移怎么办理?", "人事管理"),
        ("户口迁移怎么操作?", "人事管理"),
        ("居住证怎么办理?", "人事管理"),
        ("工作证明怎么开?", "人事管理"),
        ("收入证明怎么开?", "人事管理"),
        
        # 财务管理 (20条)
        ("工资什么时候发?", "财务管理"),
        ("年终奖怎么算?", "财务管理"),
        ("个税怎么扣除?", "财务管理"),
        ("五险一金比例是多少?", "财务管理"),
        ("住房公积金怎么提取?", "财务管理"),
        ("医保怎么报销?", "财务管理"),
        ("生育保险怎么用?", "财务管理"),
        ("工伤保险怎么申请?", "财务管理"),
        ("失业保险怎么领取?", "财务管理"),
        ("商业保险怎么理赔?", "财务管理"),
        ("工资卡怎么办理?", "财务管理"),
        ("个税专项扣除怎么填?", "财务管理"),
        ("年终奖计税方式是什么?", "财务管理"),
        ("加班费怎么算?", "财务管理"),
        ("绩效奖金怎么发放?", "财务管理"),
        ("工资条看不懂怎么办?", "财务管理"),
        ("社保缴费基数是多少?", "财务管理"),
        ("公积金贷款怎么申请?", "财务管理"),
        ("个税APP怎么操作?", "财务管理"),
        ("工资异议找谁反映?", "财务管理"),
        
        # 其他 (20条)
        ("公司食堂在哪里?", "其他"),
        ("停车位怎么申请?", "其他"),
        ("会议室怎么预定?", "其他"),
        ("公司班车时刻表?", "其他"),
        ("健身房怎么使用?", "其他"),
        ("图书室在哪里?", "其他"),
        ("茶水间在哪层?", "其他"),
        ("快递怎么收发?", "其他"),
        ("公司地址是什么?", "其他"),
        ("前台电话是多少?", "其他"),
        ("消防演习什么时候?", "其他"),
        ("团建活动有哪些?", "其他"),
        ("员工生日会有吗?", "其他"),
        ("公司周年庆活动?", "其他"),
        ("年会什么时候举办?", "其他"),
        ("培训课程有哪些?", "其他"),
        ("在线学习平台怎么用?", "其他"),
        ("职称评定怎么申请?", "其他"),
        ("公司规章制度在哪里?", "其他"),
        ("员工手册怎么获取?", "其他"),
    ]
    
    return training_data


def train_intent_classifier(
    training_data: List[Tuple[str, str]],
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Pipeline, Dict[str, float], List[str]]:
    """
    训练意图分类模型
    
    Args:
        training_data: 训练数据列表
        test_size: 测试集比例
        random_state: 随机种子
        verbose: 是否输出详细信息
        
    Returns:
        (pipeline, metrics, labels): 模型pipeline, 评估指标, 标签列表
    """
    # 分离文本和标签
    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]
    
    # 获取唯一标签列表
    unique_labels = sorted(list(set(labels)))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练数据统计")
        print(f"{'='*60}")
        print(f"总样本数: {len(texts)}")
        print(f"标签类别: {unique_labels}")
        print(f"\n各类别样本数:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  {label}: {count}条")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    if verbose:
        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)}条")
        print(f"  测试集: {len(X_test)}条")
    
    # 构建Pipeline: TF-IDF向量化 + 朴素贝叶斯分类
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=5000,      # 最大特征数
            ngram_range=(1, 2),     # 使用1-gram和2-gram
            min_df=1,               # 最小文档频率
            max_df=0.95,            # 最大文档频率
            stop_words=None,        # 不使用停用词(中文需要自定义)
            sublinear_tf=True       # 使用sublinear TF缩放
        )),
        ('classifier', MultinomialNB(
            alpha=0.1,              # 平滑参数
            fit_prior=True          # 学习先验概率
        ))
    ])
    
    # 训练模型
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始训练模型...")
        print(f"{'='*60}")
    
    pipeline.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred = pipeline.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"模型评估结果")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=unique_labels))
    
    return pipeline, metrics, unique_labels


def save_model(
    pipeline: Pipeline,
    labels: List[str],
    output_path: str,
    verbose: bool = False
) -> None:
    """
    保存模型到pickle文件
    
    Args:
        pipeline: 训练好的模型pipeline
        labels: 标签列表
        output_path: 输出路径
        verbose: 是否输出详细信息
    """
    # 提取vectorizer和classifier
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    
    # 构建模型字典
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'labels': labels
    }
    
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 序列化保存
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"模型保存成功")
        print(f"{'='*60}")
        print(f"保存路径: {output_file.absolute()}")
        print(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def load_model(model_path: str) -> Dict:
    """
    加载模型(用于验证)
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        包含vectorizer, classifier, labels的字典
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_intent(query: str, model_data: Dict) -> Tuple[str, float]:
    """
    使用加载的模型预测意图
    
    Args:
        query: 用户查询文本
        model_data: 加载的模型数据
        
    Returns:
        (predicted_label, confidence): 预测标签和置信度
    """
    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']
    
    # 向量化
    X = vectorizer.transform([query])
    
    # 预测
    predicted_label = classifier.predict(X)[0]
    
    # 获取预测概率
    probabilities = classifier.predict_proba(X)[0]
    confidence = max(probabilities)
    
    return predicted_label, confidence


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='意图分类模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认参数训练
    python train_intent_classifier.py
    
    # 指定输出路径和详细输出
    python train_intent_classifier.py --output-path ./models/intent_model.pkl --verbose
    
    # 仅指定输出路径
    python train_intent_classifier.py --output-path /path/to/model.pkl
        """
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='./intent_model.pkl',
        help='模型输出路径 (默认: ./intent_model.pkl)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='测试集比例 (默认: 0.2)'
    )
    
    args = parser.parse_args()
    
    try:
        # 生成训练数据
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"意图分类模型训练")
            print(f"{'='*60}")
        
        training_data = generate_training_data()
        
        # 训练模型
        pipeline, metrics, labels = train_intent_classifier(
            training_data,
            test_size=args.test_size,
            verbose=args.verbose
        )
        
        # 保存模型
        save_model(pipeline, labels, args.output_path, verbose=args.verbose)
        
        # 验证模型加载和预测
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"模型验证")
            print(f"{'='*60}")
            
            # 加载模型
            model_data = load_model(args.output_path)
            
            # 测试预测
            test_queries = [
                "如何报销交通费?",
                "我想请年假",
                "电脑蓝屏了",
                "入职需要什么材料?",
                "工资条在哪里看?",
                "会议室怎么预定?"
            ]
            
            print(f"\n测试预测:")
            for query in test_queries:
                label, confidence = predict_intent(query, model_data)
                print(f"  查询: {query}")
                print(f"  预测: {label} (置信度: {confidence:.4f})")
                print()
        
        # 输出最终结果
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"{'='*60}")
        print(f"模型路径: {Path(args.output_path).absolute()}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())