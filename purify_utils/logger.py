import logging


class Logger:
    def __init__(self, config, path_dir=None,adding=''):
        # https://github.com/huggingface/transformers/issues/1843#issuecomment-555598281
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

        logging.basicConfig(level=logging.DEBUG)
        self.log = logging.getLogger("log")
        if path_dir == None :
            file_handler = logging.FileHandler("{}/bert_base_mlm{}.log".format(config.model_base_path,adding))
        else:
            file_handler = logging.FileHandler("{}/bert_base_mlm{}.log".format(path_dir,adding))
        file_handler.setLevel(logging.DEBUG)
        self.log.addHandler(file_handler)


