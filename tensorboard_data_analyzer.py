import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def parse_tf_events_file(file_path):
    data = []
    for record in tf_record.tf_record_iterator(file_path):
        event = event_pb2.Event.FromString(record)
        for value in event.summary.value:
            if value.HasField('simple_value'):
                data.append((event.step, value.tag, value.simple_value))
    return data

# ログファイルのパスを指定
file_path = '/app/lego_tensorboard/nmax 14 ntensor 6 (normalized) None_1/events.out.tfevents.1701602621.218a46c2b192.2028.0'

# データの読み込みと解析
data = parse_tf_events_file(file_path)

# 解析したデータの表示（例）
#for step, tag, value in data:
    #print(f"Step: {step}, Tag: {tag}, Value: {value}")
print(data)