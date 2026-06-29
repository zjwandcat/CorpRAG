# Financial Agent API - IAM 配置
# ECS Task Execution Role、Task Role、IAM Policy

# ============================================================================
# ECS Task Execution Role
# 用于拉取 ECR 镜像和写入 CloudWatch 日志
# ============================================================================
resource "aws_iam_role" "ecs_execution" {
  name = "${local.name_prefix}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# ============================================================================
# ECS Task Execution Role Policy Attachment
# ============================================================================
resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ============================================================================
# ECS Task Execution - Secrets Manager Policy
# 允许从 Secrets Manager 读取密钥
# ============================================================================
resource "aws_iam_policy" "secrets_manager" {
  name        = "${local.name_prefix}-secrets-manager-policy"
  description = "Policy to allow ECS tasks to read secrets from Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.api_keys.arn
        ]
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "secrets_manager" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = aws_iam_policy.secrets_manager.arn
}

# ============================================================================
# ECS Task Role
# 用于应用程序运行时权限
# ============================================================================
resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# ============================================================================
# ECS Task - CloudWatch Logs Policy
# ============================================================================
resource "aws_iam_policy" "cloudwatch_logs" {
  name        = "${local.name_prefix}-cloudwatch-logs-policy"
  description = "Policy to allow ECS tasks to write to CloudWatch Logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "${aws_cloudwatch_log_group.ecs.arn}:*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "cloudwatch_logs" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.cloudwatch_logs.arn
}

# ============================================================================
# ECS Task - S3 Policy (可选)
# 用于访问 S3 存储桶（模型文件、数据文件等）
# ============================================================================
resource "aws_iam_policy" "s3_access" {
  count = var.enable_s3_access ? 1 : 0

  name        = "${local.name_prefix}-s3-access-policy"
  description = "Policy to allow ECS tasks to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "s3_access" {
  count = var.enable_s3_access ? 1 : 0

  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.s3_access[0].arn
}

# ============================================================================
# ECS Task - DynamoDB Policy (可选)
# 用于访问 DynamoDB（会话存储、缓存等）
# ============================================================================
resource "aws_iam_policy" "dynamodb_access" {
  count = var.enable_dynamodb_access ? 1 : 0

  name        = "${local.name_prefix}-dynamodb-access-policy"
  description = "Policy to allow ECS tasks to access DynamoDB"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = "arn:aws:dynamodb:${var.region}:${data.aws_caller_identity.current.account_id}:table/${var.dynamodb_table_name}"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "dynamodb_access" {
  count = var.enable_dynamodb_access ? 1 : 0

  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.dynamodb_access[0].arn
}

# ============================================================================
# ECS Task - SSM Parameter Store Policy (可选)
# 用于从 Parameter Store 读取配置
# ============================================================================
resource "aws_iam_policy" "ssm_parameters" {
  count = var.enable_ssm_parameters ? 1 : 0

  name        = "${local.name_prefix}-ssm-parameters-policy"
  description = "Policy to allow ECS tasks to read SSM parameters"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/${local.name_prefix}/*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ssm_parameters" {
  count = var.enable_ssm_parameters ? 1 : 0

  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.ssm_parameters[0].arn
}

# ============================================================================
# ECS Exec Role Policy (可选)
# 用于 ECS Exec 功能（容器内命令执行）
# ============================================================================
resource "aws_iam_role_policy_attachment" "ecs_exec" {
  count = var.enable_execute_command ? 1 : 0

  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}