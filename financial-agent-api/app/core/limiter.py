"""
限速器模块。

将 Limiter 实例放在独立模块中，避免循环导入问题。

循环导入问题说明：
- app/main.py 导入 app/api/routes_chat.py
- routes_chat.py 需要导入 limiter
- 如果 limiter 定义在 main.py 中，就会形成循环导入

解决方案：将 limiter 放在此独立模块中，两边都从这里导入。
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# 使用客户端 IP 地址作为限速的键，每个 IP 独立计数
limiter = Limiter(key_func=get_remote_address)
