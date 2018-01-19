
import classification.extractors.inception_v3_feature_extractor as inception_v3_feature_extractor
import classification.extractors.mobile_net_feature_extractor as mobile_net_feature_extractor

NAME_TO_FEATURE_EXTRACTOR = {
    'inception_v3' : inception_v3_feature_extractor.
                        InceptionV3FeatureExtractor,
    'mobile_nets' : mobile_net_feature_extractor.MobileNetFeatureExtractor
}
