# Financial Agent API - 变量定义
# 所有配置变量和敏感信息定义

# ============================================================================
# 基础配置变量
# ============================================================================
variable "region" {
  description = "AWS 区域"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "项目名称"
  type        = string
  default     = "financial-agent-api"
}

variable "environment" {
  description = "环境名称（dev、staging、prod）"
  type        = string
  default     = "dev"
}

# ============================================================================
# VPC 配置变量
# ============================================================================
variable "vpc_cidr" {
  description = "VPC CIDR 块"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "公有子网 CIDR 列表"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "私有子网 CIDR 列表"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "enable_nat_gateway" {
  description = "是否启用 NAT Gateway"
  type        = bool
  default     = true
}

# ============================================================================
# ECS 配置变量
# ============================================================================
variable "cluster_name" {
  description = "ECS 集群名称"
  type        = string
  default     = "financial-agent-cluster"
}

variable "container_name" {
  description = "容器名称"
  type        = string
  default     = "financial-agent-api"
}

variable "container_image" {
  description = "容器镜像 URL（ECR 或其他镜像仓库）"
  type        = string
}

variable "container_port" {
  description = "容器端口"
  type        = number
  default     = 8000
}

variable "health_check_port" {
  description = "健康检查端口"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "CPU 单位（Fargate: 256, 512, 1024, 2048, 4096）"
  type        = number
  default     = 1024
}

variable "memory" {
  description = "内存大小 MB（Fargate: 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192）"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "期望的任务数量"
  type        = number
  default     = 2
}

variable "enable_container_insights" {
  description = "是否启用 Container Insights"
  type        = bool
  default     = true
}

variable "enable_execute_command" {
  description = "是否启用 ECS Exec 命令"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch 日志保留天数"
  type        = number
  default     = 30
}

# ============================================================================
# Auto Scaling 配置变量
# ============================================================================
variable "enable_autoscaling" {
  description = "是否启用自动伸缩"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "最小任务数量"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "最大任务数量"
  type        = number
  default     = 10
}

variable "cpu_target_value" {
  description = "CPU 使用率目标值"
  type        = number
  default     = 70
}

variable "memory_target_value" {
  description = "内存使用率目标值"
  type        = number
  default     = 80
}

# ============================================================================
# Load Balancer 配置变量
# ============================================================================
variable "enable_https" {
  description = "是否启用 HTTPS"
  type        = bool
  default     = false
}

variable "certificate_arn" {
  description = "ACM 证书 ARN（用于 HTTPS）"
  type        = string
  default     = ""
}

variable "enable_deletion_protection" {
  description = "是否启用删除保护"
  type        = bool
  default     = false
}

# ============================================================================
# 应用环境变量
# ============================================================================
variable "nim_base_url" {
  description = "NIM 基础 URL"
  type        = string
  default     = "https://integrate.api.nvidia.com/v1"
}

variable "mlflow_tracking_uri" {
  description = "MLflow 跟踪服务器 URI"
  type        = string
  default     = "http://mlflow:5000"
}

variable "chroma_host" {
  description = "ChromaDB 主机地址"
  type        = string
  default     = "chromadb"
}

variable "chroma_port" {
  description = "ChromaDB 端口"
  type        = number
  default     = 8000
}

variable "log_level" {
  description = "日志级别"
  type        = string
  default     = "INFO"
}

# ============================================================================
# 敏感信息变量（通过环境变量或 tfvars 注入）
# ============================================================================
variable "nvidia_api_key" {
  description = "NVIDIA API Key"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "database_url" {
  description = "数据库连接 URL"
  type        = string
  sensitive   = true
  default     = ""
}

# ============================================================================
# S3 配置变量（可选）
# ============================================================================
variable "enable_s3_access" {
  description = "是否启用 S3 访问"
  type        = bool
  default     = false
}

variable "s3_bucket_name" {
  description = "S3 存储桶名称"
  type        = string
  default     = ""
}

# ============================================================================
# DynamoDB 配置变量（可选）
# ============================================================================
variable "enable_dynamodb_access" {
  description = "是否启用 DynamoDB 访问"
  type        = bool
  default     = false
}

variable "dynamodb_table_name" {
  description = "DynamoDB 表名称"
  type        = string
  default     = ""
}

# ============================================================================
# SSM Parameter Store 配置变量（可选）
# ============================================================================
variable "enable_ssm_parameters" {
  description = "是否启用 SSM Parameter Store 访问"
  type        = bool
  default     = false
}