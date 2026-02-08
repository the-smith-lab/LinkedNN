# checking over command line arguments

import os


def check_args(args):
    # avoid overwriting saved weights or other output files
    if args.train is True:
        if os.path.exists(
            args.wd + "/Train/linkedNN_" + str(args.seed) + ".weights.h5"
        ) and args.force is False:
            print("saved model with specified output name already \
                   exists. To force overwrite, use --force.")
            exit()
    if args.predict is True and args.empirical is None:
        if os.path.exists(args.wd
                          + "/Test/predictions_"
                          + str(args.seed)
                          + ".txt") and args.force is False:
            print(
                "saved predictions with specified output name already exists. \
                To force overwrite, use --force."
            )
            exit()

    # other checks
    if args.seed == 0:
        print("random number seed must be greater than 0 (and less than 2^32)")
        exit()
    if (
        args.train is False
        and args.predict is False
        and args.preprocess is False
        and args.plot_history is False
        and args.empirical is None
    ):
        print(
            "either --help, --preprocess, --train, --predict, or --plot_history"
        )
        exit()
    if args.train is True or args.predict is True or args.preprocess is True:
        if args.wd is None:
            print("specify working directory --wd")
            exit()
    if args.preprocess is True:
        if args.num_snps is None:
            print("specify num snps via --num_snps")
            exit()
        if args.n is None:
            print("specify sample size via --n")
            exit()
        if args.l is None:
            print("specify length of genome via --l")
            exit()
        if args.hold_out is None:
            print("specify num datasets to hold for testing via --hold_out")
            exit()
    if args.edge_width != "0" and args.empirical is not None:
        print(
            "can't specify edge width and empirical locations; at least not \
            currently"
        )
        exit()
