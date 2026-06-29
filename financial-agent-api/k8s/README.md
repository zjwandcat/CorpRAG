# 金融研报智能体 API — Kubernetes 部署指南

本目录包含 financial-agent-api 的完整 Kubernetes 部署清单。
**纯 CPU 架构，无 GPU 依赖**，适用于标准 K8s 集群。

---

## 前置条件

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Kubernetes | ≥ 1.27 | 集群已就绪 |
| kubectl | ≥ 1.27 | 已配置 kubeconfig |
| Metrics Server | 最新 | HPA 必需，见下方安装说明 |
| Nginx Ingress Controller | 最新 | 可选，仅 Ingress 暴露时需要 |
| 容器镜像 | — | `agent-api:latest` 已推送到集群可访问的镜像仓库 |

---

## 分步操作

### 1. 创建命名空间

```bash
kubectl apply -f namespace.yaml
```

### 2. 创建配置与密钥

```bash
# 非敏感环境变量
kubectl apply -f configmap.yaml

# 敏感信息（API Keys 等）
# 先复制示例文件，填入真实 base64 编码值
cp secret.example.yaml secret.yaml
# 编辑 secret.yaml，填入真实密钥
# 生成 base64：echo -n 'your-api-key' | base64
kubectl apply -f secret.yaml
```

> **注意**：`secret.yaml` 已加入 `.gitignore`，不会被提交到 Git。

### 3. 部署 ChromaDB 向量数据库

```bash
kubectl apply -f chromadb-deployment.yaml
kubectl apply -f chromadb-service.yaml

# 验证 ChromaDB 就绪
kubectl wait --for=condition=ready pod -l app=chromadb \
  -n agent-platform --timeout=120s
```

### 4. 部署 API 服务

```bash
kubectl apply -f api-deployment.yaml
kubectl apply -f api-service.yaml

# 验证 API Pod 就绪
kubectl wait --for=condition=ready pod -l app=agent-api \
  -n agent-platform --timeout=120s
```

### 5. 配置 HPA 自动扩缩容

```bash
# 安装 Metrics Server（如未安装）
# minikube:
minikube addons enable metrics-server
# kind / 其他集群:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 应用 HPA 配置
kubectl apply -f api-hpa.yaml
```

### 6. 配置 Ingress（可选）

```bash
# 确保 Nginx Ingress Controller 已安装
kubectl apply -f ingress.yaml

# 如需 TLS，创建证书密钥
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=agent-platform.local"
kubectl create secret tls agent-platform-tls \
  --namespace agent-platform \
  --key=tls.key --cert=tls.crt
```

### 7. 应用网络策略（可选）

```bash
# 限制 API Pod 的出站流量，仅允许 DNS、HTTPS、ChromaDB
kubectl apply -f networkpolicy.yaml
```

---

## 测试方法

### 健康检查

```bash
# Port-Forward 方式
kubectl port-forward svc/agent-api-service 8001:8001 -n agent-platform

# 测试健康端点
curl http://localhost:8001/health
```

### API 调用测试

```bash
# 使用 API Key 调用
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"query": "最近的市场趋势分析"}'
```

### SSE 流式测试

```bash
curl -N http://localhost:8001/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"query": "分析一下银行板块走势"}'
```

### Ingress 方式测试

```bash
# 添加本地 DNS 解析（测试环境）
# Windows: 编辑 C:\Windows\System32\drivers\etc\hosts
# Linux/Mac: 编辑 /etc/hosts
# 添加：INGRESS_IP  agent-platform.local

# 获取 Ingress IP
kubectl get ingress -n agent-platform

# 测试
curl -k https://agent-platform.local/health
```

---

## HPA 验证

### 查看当前 HPA 状态

```bash
kubectl get hpa -n agent-platform
```

### 查看扩缩容事件

```bash
kubectl describe hpa agent-api-hpa -n agent-platform
```

### 压测触发扩容

```bash
# 使用 hey 或 ab 进行压力测试
hey -z 2m -q 10 -H "Authorization: Bearer YOUR_API_KEY" \
  http://agent-api-service:8001/api/v1/chat

# 观察扩容
kubectl get pods -n agent-platform -w
```

### HPA 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| minReplicas | 2 | 最小副本数 |
| maxReplicas | 6 | 最大副本数 |
| CPU 阈值 | 70% | 平均 CPU 利用率超过 70% 触发扩容 |

---

## 故障排查

### Pod 无法启动

```bash
# 查看 Pod 状态
kubectl get pods -n agent-platform

# 查看 Pod 事件
kubectl describe pod <pod-name> -n agent-platform

# 查看容器日志
kubectl logs <pod-name> -n agent-platform

# 查看上一个崩溃的容器日志
kubectl logs <pod-name> -n agent-platform --previous
```

### API 返回 502/503

```bash
# 检查 readinessProbe 是否通过
kubectl describe pod <pod-name> -n agent-platform | grep -A5 Readiness

# 检查 Service 端点
kubectl get endpoints -n agent-platform

# 检查 Service 选择器是否匹配 Pod 标签
kubectl get svc agent-api-service -n agent-platform -o yaml
```

### ChromaDB 连接失败

```bash
# 检查 ChromaDB Pod 状态
kubectl get pods -l app=chromadb -n agent-platform

# 检查 ChromaDB Service
kubectl get svc chromadb-service -n agent-platform

# 从 API Pod 内部测试连接
kubectl exec -it <api-pod-name> -n agent-platform -- \
  curl -s http://chromadb-service:8000/api/v1/heartbeat
```

### HPA 无法获取指标

```bash
# 检查 Metrics Server 是否运行
kubectl get deployment metrics-server -n kube-system

# 查看 Metrics Server 日志
kubectl logs -n kube-system -l k8s-app=metrics-server

# 手动查看 Pod 资源使用
kubectl top pods -n agent-platform
```

### Ingress 无法访问

```bash
# 检查 Ingress Controller 是否运行
kubectl get pods -n ingress-nginx

# 查看 Ingress 状态
kubectl describe ingress agent-api-ingress -n agent-platform

# 检查 Ingress Controller 日志
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

### NetworkPolicy 导致连接异常

```bash
# 临时删除 NetworkPolicy 进行排查
kubectl delete networkpolicy agent-api-egress -n agent-platform

# 确认问题后重新应用
kubectl apply -f networkpolicy.yaml
```

---

## 资源清单总览

| 文件 | 资源类型 | 说明 |
|------|---------|------|
| `namespace.yaml` | Namespace | 命名空间 agent-platform |
| `configmap.yaml` | ConfigMap | 非敏感环境变量 |
| `secret.example.yaml` | Secret | 密钥示例（需复制为 secret.yaml） |
| `chromadb-deployment.yaml` | StatefulSet | ChromaDB 向量数据库 |
| `chromadb-service.yaml` | Service | ChromaDB Headless Service |
| `api-deployment.yaml` | Deployment + PVC | API 服务部署 |
| `api-service.yaml` | Service | API ClusterIP Service |
| `api-hpa.yaml` | HPA | 水平自动扩缩容 |
| `ingress.yaml` | Ingress | 外部访问入口 + TLS |
| `networkpolicy.yaml` | NetworkPolicy | Egress 出站流量限制 |