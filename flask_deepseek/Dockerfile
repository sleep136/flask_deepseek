FROM python:3.9-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_ENDPOINT=https://hf-mirror.com
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 5000

# 启动 Flask 应用
CMD ["flask", "run", "--host=0.0.0.0"]