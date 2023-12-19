# 基本イメージとしてPython 3.11を使用
FROM python:3.9.2

# 作業ディレクトリの設定
WORKDIR /usr/src/app

# 必要なPythonパッケージのインストール
RUN pip install --upgrade pip

RUN pip install --no-cache-dir numpy matplotlib scipy sympy pandas gymnasium sb3_contrib stable-baselines3[extra] tensorboard numba
# snakevizは手動でインストール
RUN pip install --no-cache-dir tensorflow snakeviz
# ポート番号の設定（必要に応じて）
# EXPOSE 8888

# コンテナ起動時のコマンド
CMD ["python"]