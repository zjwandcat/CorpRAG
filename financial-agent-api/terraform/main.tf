# Financial Agent API - Terraform 主配置
# 基础设施即代码配置入口

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # 后端配置 - 用于存储状态文件
  # 生产环境建议使用 S3 后端
  backend "local" {
    path = "terraform.tfstate"
  }
}

# AWS Provider 配置
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "financial-agent-api"
      Environment = var.environment
      ManagedBy   = "terraform"
      CreatedAt   = timestamp()
    }
  }
}

# 数据源 - 获取当前 AWS 账户信息
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# 数据源 - 获取可用区
data "aws_availability_zones" "available" {
  state = "available"
}

# 本地变量
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}