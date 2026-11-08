"""Train the missing IR-to-molecular-graph stage from scratch."""
from diffusion.training import train

if __name__ == "__main__":
    train("spectrum")
