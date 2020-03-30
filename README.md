git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/

# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim / load-env

python object_detection/builders/model_builder_test.py

To label the images run on terminal:
labelImg

Para gerar o tf_record, na linhas 32 adicionar as labels.
No terminal rodar:
load-env
python3 generate_tfrecord.py --csv_input=images/train_labels.csv --output_path=data/train.record --image_dir=images/train
python3 generate_tfrecord.py --csv_input=images/test_labels.csv --output_path=data/test.record --image_dir=images/test


Training:
Create the folder training, go to this folder and:
Copy the model config from models/research/object_detection/samples/configs
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

Faster:
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Edit the config with the same name, changing the model, etc
load-env
Run:
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

To check the progress open a new terminal and:
tensorboard --logdir='training'

Exporting the graph:
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_inception_v2_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-619 \
    --output_directory inferences