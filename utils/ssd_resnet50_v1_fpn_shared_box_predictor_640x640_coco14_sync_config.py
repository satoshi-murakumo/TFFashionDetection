"""
    config for SSD-RESNET
"""
import os


def write_config(MODEL_NAME):
    config = ("""
        model {
          ssd {
            inplace_batchnorm_update: true
            freeze_batchnorm: false
            num_classes: 90
            box_coder {
              faster_rcnn_box_coder {
                y_scale: 10.0
                x_scale: 10.0
                height_scale: 5.0
                width_scale: 5.0
              }
            }
            matcher {
              argmax_matcher {
                matched_threshold: 0.5
                unmatched_threshold: 0.5
                ignore_thresholds: false
                negatives_lower_than_unmatched: true
                force_match_for_each_row: true
                use_matmul_gather: true
              }
            }
            similarity_calculator {
              iou_similarity {
              }
            }
            encode_background_as_zeros: true
            anchor_generator {
              multiscale_anchor_generator {
                min_level: 3
                max_level: 7
                anchor_scale: 4.0
                aspect_ratios: [1.0, 2.0, 0.5]
                scales_per_octave: 2
              }
            }
            image_resizer {
              fixed_shape_resizer {
                height: 640
                width: 640
              }
            }
            box_predictor {
              weight_shared_convolutional_box_predictor {
                depth: 256
                class_prediction_bias_init: -4.6
                conv_hyperparams {
                  activation: RELU_6,
                  regularizer {
                    l2_regularizer {
                      weight: 0.0004
                    }
                  }
                  initializer {
                    random_normal_initializer {
                      stddev: 0.01
                      mean: 0.0
                    }
                  }
                  batch_norm {
                    scale: true,
                    decay: 0.997,
                    epsilon: 0.001,
                  }
                }
                num_layers_before_predictor: 4
                kernel_size: 3
              }
            }
            feature_extractor {
              type: 'ssd_resnet50_v1_fpn'
              fpn {
                min_level: 3
                max_level: 7
              }
              min_depth: 16
              depth_multiplier: 1.0
              conv_hyperparams {
                activation: RELU_6,
                regularizer {
                  l2_regularizer {
                    weight: 0.0004
                  }
                }
                initializer {
                  truncated_normal_initializer {
                    stddev: 0.03
                    mean: 0.0
                  }
                }
                batch_norm {
                  scale: true,
                  decay: 0.997,
                  epsilon: 0.001,
                }
              }
              override_base_feature_extractor_hyperparams: true
            }
            loss {
              classification_loss {
                weighted_sigmoid_focal {
                  alpha: 0.25
                  gamma: 2.0
                }
              }
              localization_loss {
                weighted_smooth_l1 {
                }
              }
              classification_weight: 1.0
              localization_weight: 1.0
            }
            normalize_loss_by_num_matches: true
            normalize_loc_loss_by_codesize: true
            post_processing {
              batch_non_max_suppression {
                score_threshold: 1e-8
                iou_threshold: 0.6
                max_detections_per_class: 100
                max_total_detections: 100
              }
              score_converter: SIGMOID
            }
          }
        }

        train_config: {
          fine_tune_checkpoint: "%(CHECKPOINT_FILE)s/model.ckpt"
          batch_size: 64
          sync_replicas: true
          startup_delay_steps: 0
          replicas_to_aggregate: 8
          num_steps: 25000
          data_augmentation_options {
            random_horizontal_flip {
            }
          }
          data_augmentation_options {
            random_crop_image {
              min_object_covered: 0.0
              min_aspect_ratio: 0.75
              max_aspect_ratio: 3.0
              min_area: 0.75
              max_area: 1.0
              overlap_thresh: 0.0
            }
          }
          optimizer {
            momentum_optimizer: {
              learning_rate: {
                cosine_decay_learning_rate {
                  learning_rate_base: .04
                  total_steps: 25000
                  warmup_learning_rate: .013333
                  warmup_steps: 2000
                }
              }
              momentum_optimizer_value: 0.9
            }
            use_moving_average: false
          }
          max_number_of_boxes: 100
          unpad_groundtruth_tensors: false
        }

        train_input_reader: {
          tf_record_input_reader {
            input_path: "%(ANNOTATIONS_DIR)s/train.record"
          }
          label_map_path: "%(ANNOTATIONS_DIR)s/label_map.pbtxt"
        }

        eval_config: {
          metrics_set: "coco_detection_metrics"
          use_moving_averages: false
          num_examples: 8000
        }

        eval_input_reader: {
          tf_record_input_reader {
            input_path: "%(ANNOTATIONS_DIR)s/test.record"
          }
          label_map_path: "%(ANNOTATIONS_DIR)s/label_map.pbtxt"
          shuffle: false
          num_readers: 1
        }
        """ % {
            'CHECKPOINT_FILE': os.path.join('/content', 'frozen_model', MODEL_NAME),
            'ANNOTATIONS_DIR': os.path.join('/content', 'data_dir', 'annotations'),
        }
    )
    config_path = os.path.join('/content', 'data_dir', 'tf_api.config')
    with open(config_path, 'w') as f:
        f.write(config)
        print("Config written to %s\nCheckpoints in %s\nTrain data from %s" % (
            config_path,
            os.path.join('/content', 'data_dir', 'checkpoints'),
            os.path.join('/content', 'data_dir', 'annotations'))
        )
