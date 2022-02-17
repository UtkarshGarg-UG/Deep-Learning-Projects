# Evaluation

* We expect the test file to be names test.txt
* Format for test.txt => -8*x*(3\*x+14)=-24*x*\*2-112\*x
* It returns the accuracy (exact match)
* Then run
    > python main.py -t


# Training

* The main file is train.py
* It contains the main training loop
* Format of train.txt expected: -8*x*(3\*x+14)=-24*x*\*2-112\*x
* For optional arguments help, use python train.py --help. Arguments have been defined at the bottom.
* The model parameters are saved in model_config.json
* You have an option to train from checkpoint using optional arguments: python main.py --checkpoint_path PATH and --resume_training True

# Attention Maps
* You can use the jupyter notebook "attention_maps" for building attention maps
* It also has code to run on your own data

# Observations
1. Solving for "a" and "c" in (ax**2 + bx + c) is much easy for the model as it requires a single step of multiplication. The hard part is solving for "b". It requires an intermediate step of multiplying then adding. Eg. (-2x + 34) * (4x - 2) = -8x**2 + 4x + 136x - 68 = -8x**2 +140x -68. So, "b" = 140
This is where the model makes the most mistakes (2.8% out of 3% test error)

2. Model fails to generalize for numbers that are out of its training range (for large positive and negative numbers). This is in line with the finding of this emnlp 2019 paper: “Do NLP Models Know Numbers? Probing Numeracy in Embeddings”
from Allen Institute for AI (AI2) et.al

3. And finally how does the model "solve" the equations? We can analyse this using attention maps. In the image, we can see 4 Attention Heads. If we look closely, we can see what the model actually "sees" to get the values.

Eg. (from validation set) To calculate the constant 8 in 8n**2, we can see the model attends to 4 (head 3) from first factor and 2 (head 2) from second.
Further, the n in 8n**2 focuses on the two n's in the factors (x-axis).
To get the sign for "+28", the model focuses (head 1) on the two negative signs of the factors 🤯! Also, interestingly, the model focuses on all the brackets of the factors for the middle numbers. I wonder why.


# Arguments in Training

arg_parser.add_argument('--seed', default=1, type=int, help='random seed')
    arg_parser.add_argument('--train_path', default='train.txt', type=str, help='raw train file')
    arg_parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
    arg_parser.add_argument('--gpu_name', default=0, type=int, help='which gpu to use')
    arg_parser.add_argument('--bs', default=512, type=int, help='batch size')
    arg_parser.add_argument('--val_frac', default=0.1, type=float, help='validation ratio')
    arg_parser.add_argument('--lr', default=0.0005, type=int, help='learning rate')
    arg_parser.add_argument('--epochs', default=150, type=int, help='num epochs')
    arg_parser.add_argument('--early_stop_count', default=5, type=int, help='early stopping count')
    arg_parser.add_argument('--checkpoint_path', default='checkpoints/checkpoint.pt', type=str, help='checkpoint path')
    arg_parser.add_argument('--resume_training', default=False, type=bool, help='resume training from checkpoint')