#!/bin/bash
set -e

# ------------------ 配置部分 ------------------
PORT_EMBED=10991
PORT_RERANK=10993
LOG_DIR=logs
mkdir -p $LOG_DIR
# ------------------------------------------------

# 启动 embedding 服务
echo "🚀 启动 embedding 服务 (端口 $PORT_EMBED)..."
nohup env SERVICE_PORT=$PORT_EMBED MODEL_TYPE=embed MODEL_CLS=BgeM3Infer MODEL_KWARGS='{"model_path": "/home/xiaogq/verify_models/bge-m3","use_fp16":false}' python src/start_server.py > $LOG_DIR/embedding.log 2>&1 &

# 等待 embedding 启动成功
echo "⏳ 等待 embedding 服务启动..."
for i in {1..60}; do
    if grep -q "Infer Server started" $LOG_DIR/embedding.log; then
        echo "✅ embedding 服务已启动成功"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "❌ embedding 启动超时，请检查 $LOG_DIR/embedding.log"
        exit 1
    fi
done

# 启动 reranker 服务
echo "🚀 启动 reranker 服务 (端口 $PORT_RERANK)..."
nohup env SERVICE_PORT=$PORT_RERANK MODEL_TYPE=rerank MODEL_CLS=BgeRerankerInfer MODEL_KWARGS='{"model_path":"/home/xiaogq/verify_models/bge-reranker-v2-m3","use_fp16":false}' python src/start_server.py > $LOG_DIR/reranker.log 2>&1 &

# 等待 reranker 启动成功
echo "⏳ 等待 reranker 服务启动..."
for i in {1..60}; do
    if grep -q "Infer Server started" $LOG_DIR/reranker.log; then
        echo "✅ reranker 服务已启动成功"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "❌ reranker 启动超时，请检查 $LOG_DIR/reranker.log"
        exit 1
    fi
done
# 等待所有服务稳定
echo "⏳ 等待服务稳定..."
sleep 5

# 启动 E2E 测试（端口号请根据需要在 test_e2e.py 中配置）
echo "🚀 运行 E2E 测试..."
python test/test_e2e.py

# 检查是否生成结果文件
if [ -f embedding.pkl ] && [ -f reranker.pkl ]; then
    echo "✅ embedding.pkl 与 reranker.pkl 已生成"
else
    echo "⚠️ 检查：未发现 embedding.pkl 或 reranker.pkl"
fi

# 运行评估
echo "📊 开始评估..."
python eval_generated_pkl.py

echo "🎉 全流程完成！"
