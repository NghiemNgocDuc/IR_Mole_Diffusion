"""Compatibility entry point for the corrected trainer; uses the new CLI."""
from diffusion.training import train

if __name__ == "__main__":
    train("diffusion")
