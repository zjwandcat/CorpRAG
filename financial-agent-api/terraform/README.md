# Financial Agent API - Terraform 基础设施配置

本目录包含 Financial Agent API 的 AWS 基础设施即代码（IaC）配置。

## 📁 文件结构

```
terraform/
├── main.tf                    # 主配置入口、Provider 配置
├── network.tf                 # VPC、Subnet、Security Group、Gateway
├── ecs.tf                     # ECS Cluster、Task Definition、Service、ALB
├── iam.tf                     # IAM Role、Policy
├── variables.tf               # 变量定义
├── outputs.tf                 # 输出定义
├── terraform.tfvars.example   # 变量配置示例
└── README.md                  # 本文档
```

## 🚀 快速开始

### 1. 前置条件

- Terraform >= 1.5.0
- AWS CLI 已配置
- 适当的 AWS IAM 权限

### 2. 配置变量

```bash
# 复制示例配置
cp terraform.tfvars.example terraform.tfvars

# 编辑配置文件
vim terraform.tfvars
```

### 3. 设置敏感信息

**推荐方式：使用环境变量**

```bash
# Linux/macOS
export TF_VAR_nvidia_api_key="nvapi-xxxx"
export TF_VAR_openai_api_key="sk-xxxx"
export TF_VAR_database_url="postgresql://user:pass@host:5432/db"

# Windows PowerShell
$env:TF_VAR_nvidia_api_key="nvapi-xxxx"
$env:TF_VAR_openai_api_key="sk-xxxx"
$env:TF_VAR_database_url="postgresql://user:pass@host:5432/db"
```

### 4. 初始化 Terraform

```bash
terraform init
```

### 5. 查看执行计划

```bash
terraform plan
```

### 6. 应用配置

```bash
terraform apply
```

### 7. 查看输出

```bash
terraform output
```

## 📋 配置说明

### 必需变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `container_image` | ECR 镜像 URL | `123456789012.dkr.ecr.us-east-1.amazonaws.com/app:latest` |
| `nvidia_api_key` | NVIDIA API Key | 从环境变量注入 |

### 可选配置

#### 启用 HTTPS

```hcl
enable_https    = true
certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/xxxx"
```

#### 启用自动伸缩

```hcl
enable_autoscaling  = true
min_capacity        = 2
max_capacity        = 10
cpu_target_value    = 70
memory_target_value = 80
```

#### 启用 S3 访问

```hcl
enable_s3_access = true
s3_bucket_name   = "your-bucket-name"
```

## 🔒 安全最佳实践

### 1. 敏感信息管理

- ✅ 使用环境变量注入 API Key
- ✅ 使用 AWS Secrets Manager 存储密钥
- ✅ 使用 `.gitignore` 排除 `terraform.tfvars`
- ❌ 不要在代码中硬编码敏感信息
- ❌ 不要将 `terraform.tfvars` 提交到版本控制

### 2. 网络安全

- VPC 使用公有和私有子网分离
- ECS 任务运行在私有子网
- ALB 部署在公有子网
- Security Group 限制访问来源

### 3. IAM 权限

- 使用最小权限原则
- Task Execution Role 仅用于拉取镜像和写日志
- Task Role 用于应用运行时权限

## 🏗️ 架构概览

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Internet Gateway                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Application Load Balancer (ALB)             │
│                  (Public Subnets)                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│               ECS Fargate Service                        │
│                  (Private Subnets)                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Task 1     │  │   Task 2     │  │   Task N     │  │
│  │ Financial    │  │ Financial    │  │ Financial    │  │
│  │ Agent API    │  │ Agent API    │  │ Agent API    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Supporting Services                         │
│  - Secrets Manager (API Keys)                            │
│  - CloudWatch Logs                                       │
│  - S3 (Optional)                                         │
│  - DynamoDB (Optional)                                   │
└─────────────────────────────────────────────────────────┘
```

## 🔧 环境变量注入

Task Definition 中自动注入的环境变量：

| 环境变量 | 来源 | 说明 |
|---------|------|------|
| `NIM_BASE_URL` | Terraform 变量 | NIM API 基础 URL |
| `MLFLOW_TRACKING_URI` | Terraform 变量 | MLflow 跟踪服务器 |
| `CHROMA_HOST` | Terraform 变量 | ChromaDB 主机 |
| `CHROMA_PORT` | Terraform 变量 | ChromaDB 端口 |
| `AWS_REGION` | Terraform 变量 | AWS 区域 |
| `NVIDIA_API_KEY` | Secrets Manager | NVIDIA API Key |
| `OPENAI_API_KEY` | Secrets Manager | OpenAI API Key |
| `DATABASE_URL` | Secrets Manager | 数据库连接 URL |

## 📊 监控与日志

### CloudWatch Logs

日志自动发送到 CloudWatch Logs：
- 日志组：`/ecs/financial-agent-api-{env}`
- 保留天数：可配置（默认 30 天）

### Container Insights

启用 Container Insights 可获得：
- CPU 和内存使用率
- 网络流量
- 任务数量

## 🔄 更新部署

### 更新容器镜像

```bash
# 修改 terraform.tfvars 中的 container_image
terraform apply
```

### 调整资源

```bash
# 修改 CPU、内存或任务数量
terraform apply
```

### 滚动更新

ECS 自动执行滚动更新，确保零停机部署。

## 🗑️ 清理资源

```bash
# 销毁所有资源
terraform destroy
```

⚠️ **警告**：这将删除所有创建的资源，包括 VPC、ECS 集群、ALB 等。

## 📝 常见问题

### Q: 如何查看应用 URL？

```bash
terraform output application_url
```

### Q: 如何查看 ECS 服务状态？

```bash
aws ecs describe-services \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --services $(terraform output -raw ecs_service_name)
```

### Q: 如何查看任务日志？

```bash
aws logs tail /ecs/financial-agent-api-prod --follow
```

### Q: 如何执行容器命令？

```bash
# 获取任务 ID
TASK_ID=$(aws ecs list-tasks \
  --cluster financial-agent-api-prod-cluster \
  --service-name financial-agent-api-prod-service \
  --query 'taskArns[0]' --output text)

# 执行命令
aws ecs execute-command \
  --cluster financial-agent-api-prod-cluster \
  --task $TASK_ID \
  --container financial-agent-api \
  --command "/bin/bash" \
  --interactive
```

## 📚 相关文档

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/)
- [AWS ECS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [AWS ALB](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)