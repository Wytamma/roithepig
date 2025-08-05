import deeplabcut
import argparse


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Run DeepLabCut analysis on a video.")
    argparser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    argparser.add_argument('--video', type=str, required=True, help='Path to the video file.')
    argparser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output results.')   
    argparser.add_argument('--cropping', type=str, default=None, help='Optional cropping coordinates in the format "x1,x2,y1,y2". If not provided, no cropping will be applied.')

    args = argparser.parse_args()

    deeplabcut.analyze_videos(
        args.config,
        [args.video],
        save_as_csv=True, 
        destfolder=args.output_dir,
        batch_size=2,
        cropping=[int(i) for i in args.cropping.split(',')] if args.cropping else None
    )
    deeplabcut.filterpredictions(
        args.config,
        args.video,
        destfolder=args.output_dir,
    )