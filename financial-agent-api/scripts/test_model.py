"""
测试意图分类模型加载和预测功能
"""

import pickle
from pathlib import Path


def test_model():
    """测试模型加载和预测"""
    # 模型路径
    model_path = Path(__file__).parent / 'intent_model.pkl'
    
    print("="*60)
    print("测试意图分类模型")
    print("="*60)
    
    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"   ✓ 模型加载成功")
    print(f"   ✓ Vectorizer类型: {type(model_data['vectorizer']).__name__}")
    print(f"   ✓ Classifier类型: {type(model_data['classifier']).__name__}")
    print(f"   ✓ 标签列表: {model_data['labels']}")
    
    # 测试预测
    print(f"\n2. 测试预测功能")
    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']
    
    test_cases = [
        "如何报销差旅费?",
        "我想请病假",
        "电脑无法连接网络",
        "入职手续怎么办?",
        "工资什么时候发放?",
        "会议室怎么预定?"
    ]
    
    for query in test_cases:
        # 向量化
        X = vectorizer.transform([query])
        # 预测
        predicted_label = classifier.predict(X)[0]
        # 获取概率
        probabilities = classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        print(f"\n   查询: {query}")
        print(f"   预测意图: {predicted_label} (置信度: {confidence:.4f})")
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    test_model()