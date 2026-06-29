# Financial Agent API - 输出定义
# Terraform 应用后的输出值

# ============================================================================
# 网络输出
# ============================================================================
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "VPC CIDR 块"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "公有子网 ID 列表"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "私有子网 ID 列表"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "NAT Gateway ID 列表"
  value       = aws_nat_gateway.main[*].id
}

# ============================================================================
# Security Group 输出
# ============================================================================
output "alb_security_group_id" {
  description = "ALB Security Group ID"
  value       = aws_security_group.alb.id
}

output "ecs_tasks_security_group_id" {
  description = "ECS Tasks Security Group ID"
  value       = aws_security_group.ecs_tasks.id
}

# ============================================================================
# ECS 输出
# ============================================================================
output "ecs_cluster_name" {
  description = "ECS 集群名称"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS 集群 ARN"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_service_name" {
  description = "ECS 服务名称"
  value       = aws_ecs_service.main.name
}

output "ecs_service_arn" {
  description = "ECS 服务 ARN"
  value       = aws_ecs_service.main.arn
}

output "task_definition_arn" {
  description = "Task Definition ARN"
  value       = aws_ecs_task_definition.main.arn
}

output "task_definition_family" {
  description = "Task Definition Family"
  value       = aws_ecs_task_definition.main.family
}

# ============================================================================
# Load Balancer 输出
# ============================================================================
output "alb_dns_name" {
  description = "ALB DNS 名称"
  value       = aws_lb.main.dns_name
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.main.arn
}

output "alb_zone_id" {
  description = "ALB Zone ID"
  value       = aws_lb.main.zone_id
}

output "target_group_arn" {
  description = "Target Group ARN"
  value       = aws_lb_target_group.main.arn
}

output "http_listener_arn" {
  description = "HTTP Listener ARN"
  value       = aws_lb_listener.http.arn
}

output "https_listener_arn" {
  description = "HTTPS Listener ARN"
  value       = var.enable_https ? aws_lb_listener.https[0].arn : ""
}

# ============================================================================
# IAM 输出
# ============================================================================
output "ecs_execution_role_arn" {
  description = "ECS Execution Role ARN"
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_task_role_arn" {
  description = "ECS Task Role ARN"
  value       = aws_iam_role.ecs_task.arn
}

# ============================================================================
# Secrets Manager 输出
# ============================================================================
output "secrets_manager_arn" {
  description = "Secrets Manager ARN"
  value       = aws_secretsmanager_secret.api_keys.arn
}

output "secrets_manager_name" {
  description = "Secrets Manager 名称"
  value       = aws_secretsmanager_secret.api_keys.name
}

# ============================================================================
# CloudWatch Logs 输出
# ============================================================================
output "cloudwatch_log_group_arn" {
  description = "CloudWatch Log Group ARN"
  value       = aws_cloudwatch_log_group.ecs.arn
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch Log Group 名称"
  value       = aws_cloudwatch_log_group.ecs.name
}

# ============================================================================
# 应用访问信息
# ============================================================================
output "application_url" {
  description = "应用访问 URL（HTTP）"
  value       = "http://${aws_lb.main.dns_name}"
}

output "application_url_https" {
  description = "应用访问 URL（HTTPS）"
  value       = var.enable_https ? "https://${aws_lb.main.dns_name}" : ""
}

# ============================================================================
# Auto Scaling 输出
# ============================================================================
output "autoscaling_target_id" {
  description = "Auto Scaling Target ID"
  value       = var.enable_autoscaling ? aws_appautoscaling_target.ecs[0].resource_id : ""
}

# ============================================================================
# 综合信息输出
# ============================================================================
output "deployment_info" {
  description = "部署综合信息"
  value = {
    environment       = var.environment
    region           = var.region
    cluster_name     = aws_ecs_cluster.main.name
    service_name     = aws_ecs_service.main.name
    alb_dns_name     = aws_lb.main.dns_name
    application_url  = "http://${aws_lb.main.dns_name}"
    container_image  = var.container_image
    desired_count    = var.desired_count
    cpu              = var.cpu
    memory           = var.memory
  }
}