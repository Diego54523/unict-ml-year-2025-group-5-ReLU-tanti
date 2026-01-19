import argparse

def main():
    parser = argparse.ArgumentParser(
        description = "MRI Classification - Training & Testing Pipeline"
    )

    parser.add_argument(
        "--task",
        type = str,
        required = True,
        choices = [
            "train_radnet",
            "train_cnn", 
            "train_mlp", 
            "train_mlp_on_radnet",
            "extract_resnet",
            "extract_custom",
            "extract_radnet"
        ],
        help = "Task da eseguire"
    )

    args = parser.parse_args()

    if args.task == "train_radnet":
        from training.radnet import RadNetRunner
        RadNetRunner().run()

    elif args.task == "train_cnn":
        from training.Training_Custom_CNN import main as run
        run()
        
    elif args.task == "train_mlp":
        from training.MLP import main as run
        run()
        
    elif args.task == "train_mlp_on_radnet":
        from training.train_mlp_on_radnet_features import main as run
        run()

    elif args.task == "extract_resnet":
        from training.Feature_extraction_ResNet18 import main as run
        run()
        
    elif args.task == "extract_custom":
        from training.Extract_Features_Custom_CNN import main as run
        run()
        
    elif args.task == "extract_radnet":
        from training.extract_feature_radnet import main as run
        run()

    else:
        raise ValueError("Task non riconosciuto")


if __name__ == "__main__":
    main()