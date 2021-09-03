from audio_preprocessing.FullSubNet.audio_zen.utils import initialize_module


def denoise(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Inference")
#     parser.add_argument("-C", "--configuration", type=str, required=True, help="Config file.")
#     parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="The path of the model's checkpoint.")
#     parser.add_argument("-O", "--output_dir", type=str, required=True, help="The path for saving enhanced speeches.")
#     args = parser.parse_args()
#
#     configuration = toml.load(args.configuration)
#     checkpoint_path = args.model_checkpoint_path
#     output_dir = args.output_dir
#
#     main(configuration, checkpoint_path, output_dir)
