# classification
TensorFlow Classification API
  Modeled after TensorFlow Object Detection API

## Adding you own model
  
  1) add your new model file to extractors (see inception_v3 example)
  2) have your model implement all the required methods and inherit FeatureExtractor
  3) add your model to builders/feature_extractor_map.py (same as inception)

## Usage
  1) Create a config file, similar to the example pipeline_conifg.config
  2) build protocol buffers with `protoc classification/protos/*.proto --python_out=.
` from outside classification dir
  3) cd in to classification directory and run ``export PYTHONPATH=`pwd`:$PYTHONPATH ``
  4) from outside classification dir run `python3 classification/trainer.py --pipeline_config {path_to_pipeline_config}`

## Data Format
  This API utilizes TFRecord Format. There is a serializer in serializers that expects your data in the following format.
  
          inputs_dir/images ->
                    {image_name}.{image_file_extension}
          inputs_dir/labels/{image_name}.json:
                  {
                      {annotation_1} : {category_name},
                      {annotation_2} : {category_name},
                      ...
                  }
           inputs_dir/annotations.json (mapping between string label and number) :
                  {
                      {annotation_1} : {category_name: category_byte, ...},
                      {annotation_2} : {category_name: category_byte, ...},
                      ...
                  }
