import torch
from Generator import Generator

def test_generator_shapes():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator().to(device)
    generator.eval()

    test_cases = [
        (1,  3, 64, 64),   # single image
        (4,  3, 64, 64),   # small batch
        (32, 3, 64, 64),   # train batch
        (256,3, 64, 64),   # full batch
    ]

    print(f"{'Input Shape':<30} {'Output Shape':<30} {'Pass'}")
    print("-" * 70)

    with torch.no_grad():
        for shape in test_cases:
            x = torch.randn(*shape).to(device)
            out = generator(x)
            expected = (shape[0], 3, 125, 125)
            passed = tuple(out.shape) == expected
            print(f"{str(tuple(x.shape)):<30} {str(tuple(out.shape)):<30} {'✓' if passed else f'✗ expected {expected}'}")

if __name__ == "__main__":
    test_generator_shapes()