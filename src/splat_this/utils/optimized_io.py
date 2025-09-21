"""Optimized I/O operations for SplatThis."""

import os
import mmap
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, BinaryIO
import logging

from ..utils.profiler import global_profiler

logger = logging.getLogger(__name__)


class OptimizedFileWriter:
    """Optimized file writer with buffering and atomic operations."""

    def __init__(self, buffer_size: int = 64 * 1024):  # 64KB default buffer
        self.buffer_size = buffer_size

    @global_profiler.profile_function("atomic_file_write")
    def write_atomic(self, content: str, target_path: Path, encoding: str = "utf-8") -> None:
        """Write file atomically using temporary file."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Use temporary file in same directory for atomic operation
        temp_dir = target_path.parent

        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding=encoding,
            dir=temp_dir,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            temp_path = Path(temp_file.name)

            try:
                # Write with buffering
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk

                # Atomic move
                shutil.move(str(temp_path), str(target_path))
                logger.debug(f"Atomically wrote {len(content)} bytes to {target_path}")

            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise e

    @global_profiler.profile_function("buffered_file_write")
    def write_buffered(self, content: str, target_path: Path, encoding: str = "utf-8") -> None:
        """Write file with optimized buffering."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, 'w', encoding=encoding, buffering=self.buffer_size) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        logger.debug(f"Wrote {len(content)} bytes to {target_path} with buffering")


class OptimizedFileReader:
    """Optimized file reader with memory mapping for large files."""

    @global_profiler.profile_function("optimized_file_read")
    def read_file(self, file_path: Path, encoding: str = "utf-8") -> str:
        """Read file with optimization based on size."""
        file_path = Path(file_path)
        file_size = file_path.stat().st_size

        # Use memory mapping for large files (>1MB)
        if file_size > 1024 * 1024:
            return self._read_with_mmap(file_path, encoding)
        else:
            return self._read_standard(file_path, encoding)

    def _read_with_mmap(self, file_path: Path, encoding: str) -> str:
        """Read large file using memory mapping."""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                return mmapped_file.read().decode(encoding)

    def _read_standard(self, file_path: Path, encoding: str) -> str:
        """Read small file using standard method."""
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()


class StreamingWriter:
    """Streaming writer for very large content."""

    def __init__(self, buffer_size: int = 128 * 1024):  # 128KB buffer
        self.buffer_size = buffer_size

    @global_profiler.profile_function("streaming_write")
    def write_streaming(
        self,
        content_generator,
        target_path: Path,
        encoding: str = "utf-8"
    ) -> None:
        """Write content from generator in streaming fashion."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        bytes_written = 0

        with open(target_path, 'w', encoding=encoding, buffering=self.buffer_size) as f:
            for chunk in content_generator:
                f.write(chunk)
                bytes_written += len(chunk)

                # Periodic flush for very large files
                if bytes_written % (1024 * 1024) == 0:  # Every 1MB
                    f.flush()

            f.flush()
            os.fsync(f.fileno())

        logger.info(f"Streamed {bytes_written} bytes to {target_path}")


class AsyncFileProcessor:
    """Asynchronous file processor for batch operations."""

    def __init__(self):
        self.writer = OptimizedFileWriter()
        self.reader = OptimizedFileReader()

    @global_profiler.profile_function("batch_file_operations")
    def batch_write(self, files_data: list, base_path: Path) -> None:
        """Write multiple files efficiently."""
        base_path = Path(base_path)

        for file_info in files_data:
            file_path = base_path / file_info['path']
            content = file_info['content']

            # Use atomic write for safety
            self.writer.write_atomic(content, file_path)

        logger.info(f"Batch wrote {len(files_data)} files to {base_path}")


# Convenience functions for common operations
def write_svg_optimized(content: str, output_path: Path) -> None:
    """Optimized SVG writing with appropriate method based on size."""
    content_size = len(content)

    writer = OptimizedFileWriter()

    if content_size > 10 * 1024 * 1024:  # >10MB
        logger.info("Using atomic write for large SVG file")
        writer.write_atomic(content, output_path)
    else:
        logger.info("Using buffered write for SVG file")
        writer.write_buffered(content, output_path)


def estimate_write_time(content_size: int) -> float:
    """Estimate file write time based on content size."""
    # Rough estimates based on typical SSD performance
    base_time = 0.001  # 1ms base overhead

    if content_size < 1024 * 1024:  # <1MB
        write_speed = 500 * 1024 * 1024  # 500MB/s
    elif content_size < 10 * 1024 * 1024:  # <10MB
        write_speed = 300 * 1024 * 1024  # 300MB/s
    else:  # >10MB
        write_speed = 200 * 1024 * 1024  # 200MB/s

    return base_time + (content_size / write_speed)


def get_optimal_buffer_size(file_size: int) -> int:
    """Get optimal buffer size based on file size."""
    if file_size < 1024 * 1024:  # <1MB
        return 8 * 1024  # 8KB
    elif file_size < 10 * 1024 * 1024:  # <10MB
        return 64 * 1024  # 64KB
    elif file_size < 100 * 1024 * 1024:  # <100MB
        return 256 * 1024  # 256KB
    else:
        return 1024 * 1024  # 1MB