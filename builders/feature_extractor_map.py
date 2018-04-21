
import classification.extractors.inception_v3_feature_extractor as inception_v3_feature_extractor
import classification.extractors.mobile_net_feature_extractor as mobile_net_feature_extractor
import classification.extractors.inception_v4_feature_extractor as inception_v4_feature_extractor
import classification.extractors.resnet_v2_50_feature_extractor as resnet_v2_50_feature_extractor
import classification.extractors.nasnet_mobile_feature_extractor as nasnet_mobile_feature_extractor
import classification.extractors.mobilenet_v2_feature_extractor as mobilenet_v2_feature_extractor

NAME_TO_FEATURE_EXTRACTOR = {
    'inception_v3': inception_v3_feature_extractor.
                        InceptionV3FeatureExtractor,
    'mobile_nets' : mobile_net_feature_extractor.MobileNetFeatureExtractor,
    'inception_v4': inception_v4_feature_extractor.
                        InceptionV4FeatureExtractor,
    'resnet_v2_50'   : resnet_v2_50_feature_extractor.
                        ResNetV250FeatureExtractor,
    'nasnet_mobile' : nasnet_mobile_feature_extractor.
                        NASNetMobileFeatureExtractor,
    'mobilenet_v2' : mobilenet_v2_feature_extractor.MobileNetV2FeatureExtractor,
}
