from reid.evaluating.frame_evaluator import FrameEvaluator

__data_factory = {
    'market': FrameEvaluator,
    'market_sct_tran': FrameEvaluator,
    'duke_sct_tran': FrameEvaluator,
    'msmt': FrameEvaluator,
    'duke': FrameEvaluator,
}


def init_evaluator(name, model, flip):
    if name not in __data_factory.keys():
        raise KeyError("Unknown data_components: {}".format(name))
    return __data_factory[name](model, flip)
