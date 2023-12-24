# 基本イメージとしてPython 3.11を使用
FROM python:3.9

# 作業ディレクトリの設定
WORKDIR /usr/src/app

# 必要なPythonパッケージのインストール
RUN pip install --upgrade pip

RUN pip install --no-cache-dir numpy scipy sympy gymnasium sb3_contrib tensorboard
#RUN pip install --no-cache-dir stable-baselines3[extra]
#RUN pip install --no-cache-dir matplotlib pandas tensorflow snakeviz numba
# ポート番号の設定（必要に応じて）
# EXPOSE 8888

# コンテナ起動時のコマンド
CMD ["python"]