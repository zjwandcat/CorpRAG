#!/usr/bin/env python3
"""
Terraform 配置验证脚本
验证配置文件的完整性和安全性
"""

import os
import re
from pathlib import Path


def check_hardcoded_secrets(content: str, filename: str) -> list:
    """检查是否硬编码了敏感信息"""
    issues = []
    
    # 检查硬编码的 API Key 模式
    patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI API Key'),
        (r'nvapi-[a-zA-Z0-9]{20,}', 'NVIDIA API Key'),
        (r'password\s*=\s*["\'][^"\']+["\']', '硬编码密码'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', '硬编码 API Key'),
    ]
    
    for pattern, desc in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append(f"❌ {filename}: 发现 {desc}")
    
    return issues


def check_sensitive_variables(content: str, filename: str) -> list:
    """检查敏感变量是否标记为 sensitive"""
    issues = []
    
    # 查找变量定义
    var_pattern = r'variable\s+"([^"]+)"\s+\{([^}]+)\}'
    matches = re.findall(var_pattern, content, re.DOTALL)
    
    sensitive_keywords = ['api_key', 'secret', 'password', 'token', 'credential']
    
    for var_name, var_body in matches:
        is_sensitive_name = any(kw in var_name.lower() for kw in sensitive_keywords)
        has_sensitive_attr = 'sensitive' in var_body
        
        if is_sensitive_name and not has_sensitive_attr:
            issues.append(f"⚠️  {filename}: 变量 '{var_name}' 应标记为 sensitive = true")
    
    return issues


def validate_terraform_files(directory: str) -> dict:
    """验证所有 Terraform 配置文件"""
    results = {
        'files_checked': 0,
        'issues': [],
        'warnings': [],
        'passed': []
    }
    
    tf_files = list(Path(directory).glob('*.tf'))
    
    for tf_file in tf_files:
        results['files_checked'] += 1
        content = tf_file.read_text(encoding='utf-8')
        
        # 检查硬编码敏感信息
        issues = check_hardcoded_secrets(content, tf_file.name)
        results['issues'].extend(issues)
        
        # 检查敏感变量标记
        warnings = check_sensitive_variables(content, tf_file.name)
        results['warnings'].extend(warnings)
        
        if not issues and not warnings:
            results['passed'].append(f"✅ {tf_file.name}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Terraform 配置验证")
    print("=" * 60)
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    terraform_dir = script_dir
    
    print(f"\n📁 验证目录: {terraform_dir}\n")
    
    # 验证配置文件
    results = validate_terraform_files(terraform_dir)
    
    # 输出结果
    print(f"🔍 检查了 {results['files_checked']} 个配置文件\n")
    
    # 输出通过的文件
    if results['passed']:
        print("✅ 通过验证的文件:")
        for msg in results['passed']:
            print(f"  {msg}")
        print()
    
    # 输出警告
    if results['warnings']:
        print("⚠️  警告:")
        for msg in results['warnings']:
            print(f"  {msg}")
        print()
    
    # 输出问题
    if results['issues']:
        print("❌ 发现问题:")
        for msg in results['issues']:
            print(f"  {msg}")
        print()
        return 1
    
    # 检查必需文件
    required_files = [
        'main.tf', 'network.tf', 'ecs.tf', 
        'iam.tf', 'variables.tf', 'outputs.tf'
    ]
    
    missing_files = []
    for req_file in required_files:
        if not (terraform_dir / req_file).exists():
            missing_files.append(req_file)
    
    if missing_files:
        print("❌ 缺少必需文件:")
        for f in missing_files:
            print(f"  - {f}")
        print()
        return 1
    
    print("=" * 60)
    print("✅ 所有验证通过！")
    print("=" * 60)
    print("\n📋 下一步:")
    print("  1. 复制 terraform.tfvars.example 为 terraform.tfvars")
    print("  2. 设置环境变量: export TF_VAR_nvidia_api_key='your-key'")
    print("  3. 运行: terraform init")
    print("  4. 运行: terraform plan")
    print("  5. 运行: terraform apply")
    print()
    
    return 0


if __name__ == '__main__':
    exit(main())