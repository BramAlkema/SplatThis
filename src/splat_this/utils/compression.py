"""Ultra-compact splat compression utilities."""

import struct
import zlib
from typing import List, Tuple
import math as mathlib

if False:  # TYPE_CHECKING
    from ..core.extract import Gaussian
else:
    from ..core.extract import Gaussian


class CompactSplatEncoder:
    """Encode splats in ultra-compact binary format."""

    def __init__(self, quantization_bits: int = 12):
        """Initialize encoder with quantization level.

        Args:
            quantization_bits: Bits for position/size quantization (8-16)
        """
        self.quantization_bits = quantization_bits
        self.max_val = (1 << quantization_bits) - 1

    def encode_splats(self, splats: List["Gaussian"], width: int, height: int) -> bytes:
        """Encode splats to ultra-compact binary format.

        Format per splat (11 bytes):
        - x, y: 12 bits each (quantized to image dimensions)
        - rx, ry: 8 bits each (quantized to max size)
        - theta: 8 bits (quantized to 0-255 for 0-2π)
        - r, g, b: 8 bits each
        - alpha: 4 bits (16 levels)

        Total: 12+12+8+8+8+8+8+8+4 = 84 bits = 10.5 bytes per splat
        """
        if not splats:
            return b''

        # Calculate quantization scales
        x_scale = self.max_val / width
        y_scale = self.max_val / height
        size_scale = 255 / max(max(s.rx, s.ry) for s in splats)

        encoded_data = bytearray()

        for splat in splats:
            # Quantize position (12 bits each)
            x_q = int(splat.x * x_scale) & self.max_val
            y_q = int(splat.y * y_scale) & self.max_val

            # Quantize size (8 bits each)
            rx_q = int(splat.rx * size_scale) & 0xFF
            ry_q = int(splat.ry * size_scale) & 0xFF

            # Quantize rotation (8 bits for 0-2π)
            theta_q = int((splat.theta % (2 * mathlib.pi)) * 255 / (2 * mathlib.pi)) & 0xFF

            # Colors are already 8-bit
            r, g, b = splat.r & 0xFF, splat.g & 0xFF, splat.b & 0xFF

            # Alpha to 4 bits (16 levels)
            alpha_q = int(splat.a * 15) & 0x0F

            # Pack into bytes (11 bytes per splat)
            # Bytes 0-2: x (12 bits) + y (12 bits) = 24 bits = 3 bytes
            pos_packed = (x_q << 12) | y_q
            encoded_data.extend(struct.pack('>I', pos_packed)[1:])  # Skip first byte

            # Bytes 3-6: rx, ry, theta, r
            encoded_data.extend(struct.pack('BBBB', rx_q, ry_q, theta_q, r))

            # Bytes 7-8: g, b
            encoded_data.extend(struct.pack('BB', g, b))

            # Byte 9: alpha (4 bits) + reserved (4 bits)
            encoded_data.append(alpha_q << 4)

        # Add header with metadata
        header = struct.pack('>IIHHHH',
                            len(splats),      # Number of splats
                            len(encoded_data), # Data length
                            width,            # Image width
                            height,           # Image height
                            self.quantization_bits, # Quantization level
                            int(size_scale * 1000)  # Size scale * 1000
                            )

        return header + bytes(encoded_data)

    def compress_splats(self, splats: List["Gaussian"], width: int, height: int) -> bytes:
        """Encode and compress splats with zlib.

        Returns:
            Compressed binary data
        """
        encoded = self.encode_splats(splats, width, height)
        compressed = zlib.compress(encoded, level=9)

        # Add compression header
        comp_header = struct.pack('>II', len(encoded), len(compressed))
        return comp_header + compressed


class CompactSplatDecoder:
    """Decode ultra-compact splat format."""

    def decode_splats(self, data: bytes) -> Tuple[List["Gaussian"], int, int]:
        """Decode splats from binary format.

        Returns:
            Tuple of (splats, width, height)
        """
        if len(data) < 16:
            return [], 0, 0

        # Read header
        header = struct.unpack('>IIHHHH', data[:16])
        num_splats, data_len, width, height, quant_bits, size_scale_int = header
        size_scale = size_scale_int / 1000.0

        if len(data) < 16 + data_len:
            raise ValueError("Incomplete splat data")

        # Calculate scales
        max_val = (1 << quant_bits) - 1
        x_scale = width / max_val
        y_scale = height / max_val

        splats = []
        offset = 16

        for i in range(num_splats):
            if offset + 10 > len(data):
                break

            # Unpack position (3 bytes = 24 bits)
            pos_bytes = b'\x00' + data[offset:offset+3]
            pos_packed = struct.unpack('>I', pos_bytes)[0]
            x_q = (pos_packed >> 12) & max_val
            y_q = pos_packed & max_val

            # Unpack other data
            rx_q, ry_q, theta_q, r = struct.unpack('BBBB', data[offset+3:offset+7])
            g, b = struct.unpack('BB', data[offset+7:offset+9])
            alpha_packed = data[offset+9]
            alpha_q = (alpha_packed >> 4) & 0x0F

            # Dequantize
            x = x_q * x_scale
            y = y_q * y_scale
            rx = rx_q / size_scale
            ry = ry_q / size_scale
            theta = theta_q * 2 * mathlib.pi / 255
            alpha = alpha_q / 15.0

            splat = Gaussian(
                x=x, y=y, rx=rx, ry=ry, theta=theta,
                r=r, g=g, b=b, a=alpha
            )
            splats.append(splat)

            offset += 10

        return splats, width, height

    def decompress_splats(self, compressed_data: bytes) -> Tuple[List["Gaussian"], int, int]:
        """Decompress and decode splats.

        Returns:
            Tuple of (splats, width, height)
        """
        if len(compressed_data) < 8:
            return [], 0, 0

        # Read compression header
        orig_len, comp_len = struct.unpack('>II', compressed_data[:8])

        # Decompress
        compressed = compressed_data[8:8+comp_len]
        decompressed = zlib.decompress(compressed)

        if len(decompressed) != orig_len:
            raise ValueError("Decompression failed")

        return self.decode_splats(decompressed)


def analyze_compression_ratio(splats: List["Gaussian"], width: int, height: int) -> dict:
    """Analyze compression ratios for different formats."""

    # Original bitmap size
    bitmap_size = width * height * 3  # RGB24

    # Current SVG size estimate (text format)
    svg_size_estimate = len(splats) * 150  # ~150 chars per splat in SVG

    # Compact binary format
    encoder = CompactSplatEncoder()
    compact_binary = encoder.encode_splats(splats, width, height)

    # Compressed format
    compressed_binary = encoder.compress_splats(splats, width, height)

    return {
        'bitmap_bytes': bitmap_size,
        'svg_estimate_bytes': svg_size_estimate,
        'compact_binary_bytes': len(compact_binary),
        'compressed_binary_bytes': len(compressed_binary),
        'svg_compression_ratio': bitmap_size / svg_size_estimate,
        'binary_compression_ratio': bitmap_size / len(compact_binary),
        'compressed_compression_ratio': bitmap_size / len(compressed_binary),
        'vs_svg_improvement': svg_size_estimate / len(compressed_binary),
    }


# Example usage and benchmarking
if __name__ == "__main__":
    # Simulate some splats
    test_splats = []
    for i in range(1500):
        splat = Gaussian(
            x=float(i % 1920),
            y=float(i % 1080),
            rx=float(5 + i % 20),
            ry=float(5 + i % 20),
            theta=float(i % 628) / 100,  # 0-2π
            r=i % 256,
            g=(i * 2) % 256,
            b=(i * 3) % 256,
            a=0.5 + (i % 100) / 200
        )
        test_splats.append(splat)

    # Analyze compression
    results = analyze_compression_ratio(test_splats, 1920, 1080)

    print("Compression Analysis:")
    print(f"Original bitmap: {results['bitmap_bytes']:,} bytes")
    print(f"SVG text format: {results['svg_estimate_bytes']:,} bytes")
    print(f"Compact binary: {results['compact_binary_bytes']:,} bytes")
    print(f"Compressed binary: {results['compressed_binary_bytes']:,} bytes")
    print()
    print(f"SVG vs bitmap: {results['svg_compression_ratio']:.1f}:1")
    print(f"Binary vs bitmap: {results['binary_compression_ratio']:.1f}:1")
    print(f"Compressed vs bitmap: {results['compressed_compression_ratio']:.1f}:1")
    print(f"Compressed vs SVG: {results['vs_svg_improvement']:.1f}:1")