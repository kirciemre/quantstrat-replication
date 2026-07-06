# quantstrat-replication

This is our replication project for Deep reinforcement learning for optimal trading with partial information.

You can start full test of paper with this command.

python main.py --full

It will start tarining first and then run synthetic repication part.

It may take a bit long depends on your GPU and it will designed to use accelerator mps for Apple Silicon or CUDA for Nvidia GPU.

You can validate code, tables and models with this code;

python main.py --steps 10 --pretrain_steps 10 --test_episodes 5 --test_steps 10

It should be faster than full testing code.
