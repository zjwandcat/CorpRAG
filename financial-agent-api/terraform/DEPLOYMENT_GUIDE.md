# Financial Agent API - Terraform 部署指南

## 📋 前置条件

### 1. 安装 Terraform

**Windows:**
```powershell
# 使用 Chocolatey
choco install terraform

# 或手动下载
# https://www.terraform.io/downloads
```

**Linux/macOS:**
```bash
# 使用 tfenv（推荐）
brew install tfenv
tfenv install 1.5.0

# 或直接下载
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
unzip terraform_1.5.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
```

### 2. 配置 AWS CLI

```bash
# 安装 AWS CLI
pip install awscli

# 配置凭证
aws configure
# 输入 Access Key ID、Secret Access Key、默认区域

# 验证配置
aws sts get-caller-identity
```

### 3. 准备 ECR 镜像

```bash
# 创建 ECR 仓库
aws ecr create-repository --repository-name financial-agent-api

# 登录 ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# 构建并推送镜像
docker build -t financial-agent-api .
docker tag financial-agent-api:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/financial-agent-api:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/financial-agent-api:latest
```

## 🚀 部署步骤

### 步骤 1: 配置变量

```bash
cd financial-agent-api/terraform

# 复制示例配置
cp terraform.tfvars.example terraform.tfvars

# 编辑配置文件
vim terraform.tfvars
```

**必需配置项：**
```hcl
container_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/financial-agent-api:latest"
```

### 步骤 2: 设置敏感信息

**方式 1: 环境变量（推荐）**

```bash
# Linux/macOS
export TF_VAR_nvidia_api_key="nvapi-xxxx-xxxx-xxxx"
export TF_VAR_openai_api_key="sk-xxxx-xxxx-xxxx"
export TF_VAR_database_url="postgresql://user:pass@host:5432/db"

# Windows PowerShell
$env:TF_VAR_nvidia_api_key="nvapi-xxxx-xxxx-xxxx"
$env:TF_VAR_openai_api_key="sk-xxxx-xxxx-xxxx"
$env:TF_VAR_database_url="postgresql://user:pass@host:5432/db"
```

**方式 2: terraform.tfvars（不推荐提交到 Git）**

```hcl
nvidia_api_key = "nvapi-xxxx-xxxx-xxxx"
openai_api_key = "sk-xxxx-xxxx-xxxx"
database_url   = "postgresql://user:pass@host:5432/db"
```

### 步骤 3: 初始化 Terraform

```bash
terraform init
```

**输出示例：**
```
Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/aws versions matching "~> 5.0"...
- Installing hashicorp/aws v5.31.0...
Terraform has been successfully initialized!
```

### 步骤 4: 查看执行计划

```bash
terraform plan
```

**输出示例：**
```
Terraform will perform the following actions:

  # aws_vpc.main will be created
  + resource "aws_vpc" "main" {
      + cidr_block           = "10.0.0.0/16"
      + enable_dns_hostnames = true
      ...
    }

  # aws_ecs_cluster.main will be created
  + resource "aws_ecs_cluster" "main" {
      + name = "financial-agent-api-prod-cluster"
      ...
    }

Plan: 45 to add, 0 to change, 0 to destroy.
```

### 步骤 5: 应用配置

```bash
# 自动确认
terraform apply -auto-approve

# 或手动确认
terraform apply
# 输入 yes 确认
```

### 步骤 6: 查看输出

```bash
# 查看所有输出
terraform output

# 查看特定输出
terraform output application_url
terraform output ecs_cluster_name
```

## 📊 验证部署

### 1. 检查 ECS 服务状态

```bash
# 获取集群和服务名称
CLUSTER=$(terraform output -raw ecs_cluster_name)
SERVICE=$(terraform output -raw ecs_service_name)

# 查看服务状态
aws ecs describe-services \
  --cluster $CLUSTER \
  --services $SERVICE \
  --query 'services[0].[status,runningCount,desiredCount]'
```

### 2. 检查任务健康状态

```bash
# 列出任务
aws ecs list-tasks \
  --cluster $CLUSTER \
  --service-name $SERVICE

# 查看任务详情
aws ecs describe-tasks \
  --cluster $CLUSTER \
  --tasks <task-arn>
```

### 3. 测试应用访问

```bash
# 获取 ALB URL
ALB_URL=$(terraform output -raw application_url)

# 健康检查
curl $ALB_URL/health

# API 测试
curl -X POST $ALB_URL/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}'
```

### 4. 查看日志

```bash
# 实时查看日志
aws logs tail /ecs/financial-agent-api-prod --follow

# 查看历史日志
aws logs get-log-events \
  --log-group-name /ecs/financial-agent-api-prod \
  --log-stream-name <stream-name>
```

## 🔄 更新部署

### 更新容器镜像

```bash
# 1. 推送新镜像到 ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/financial-agent-api:v2

# 2. 更新 terraform.tfvars
container_image = "...financial-agent-api:v2"

# 3. 应用更新
terraform apply
```

### 调整资源

```bash
# 修改 terraform.tfvars
cpu    = 2048
memory = 4096

# 应用更新
terraform apply
```

### 扩缩容

```bash
# 手动调整任务数量
desired_count = 5

# 或启用自动伸缩
enable_autoscaling = true
min_capacity       = 2
max_capacity       = 10
```

## 🗑️ 清理资源

### 删除所有资源

```bash
# 查看将删除的资源
terraform plan -destroy

# 确认删除
terraform destroy
# 输入 yes 确认
```

⚠️ **警告**：这将删除所有创建的资源！

### 部分删除

```bash
# 删除特定资源（不推荐）
terraform destroy -target aws_ecs_service.main
```

## 🔧 故障排查

### 问题 1: 任务无法启动

**检查：**
```bash
# 查看任务停止原因
aws ecs describe-tasks \
  --cluster $CLUSTER \
  --tasks <task-arn> \
  --query 'tasks[0].stoppedReason'
```

**常见原因：**
- 镜像拉取失败：检查 ECR 权限
- 内存不足：增加 memory 配置
- 健康检查失败：检查应用启动

### 问题 2: 服务无法访问

**检查：**
```bash
# 检查 Target Group 健康状态
aws elbv2 describe-target-health \
  --target-group-arn $(terraform output -raw target_group_arn)

# 检查 Security Group
aws ec2 describe-security-groups \
  --group-ids $(terraform output -raw alb_security_group_id)
```

### 问题 3: 密钥无法读取

**检查：**
```bash
# 验证 Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id $(terraform output -raw secrets_manager_name)

# 检查 IAM 权限
aws iam get-role-policy \
  --role-name financial-agent-api-prod-ecs-execution-role \
  --policy-name secrets-manager-policy
```

## 📈 监控与告警

### CloudWatch Dashboard

```bash
# 创建 Dashboard
aws cloudwatch put-dashboard \
  --dashboard-name financial-agent-api \
  --dashboard-body file://dashboard.json
```

### 设置告警

```bash
# CPU 使用率告警
aws cloudwatch put-metric-alarm \
  --alarm-name high-cpu \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## 🔐 安全最佳实践

### 1. 启用 HTTPS

```hcl
enable_https    = true
certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/xxxx"
```

### 2. 限制访问

```hcl
# 在 network.tf 中修改 Security Group
ingress {
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["10.0.0.0/8"]  # 仅允许内网访问
}
```

### 3. 启用删除保护

```hcl
enable_deletion_protection = true
```

### 4. 使用 S3 后端存储状态

```hcl
terraform {
  backend "s3" {
    bucket         = "terraform-state"
    key            = "financial-agent-api/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## 📚 参考资料

- [Terraform AWS Provider 文档](https://registry.terraform.io/providers/hashicorp/aws/)
- [AWS ECS Fargate 开发指南](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [AWS ALB 用户指南](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
- [Terraform 最佳实践](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)