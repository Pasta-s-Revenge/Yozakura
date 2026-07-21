# Build memory

The legacy builder loads base and target models simultaneously and applies a full dense SVD. For multi-billion parameter checkpoints this can exceed Colab RAM/VRAM even at low rank. Use a single-model streaming builder before treating 4B+ Colab builds as supported.
