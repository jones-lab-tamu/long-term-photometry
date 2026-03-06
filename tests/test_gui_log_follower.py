"""
Unit tests for LogFollower PURE tail logic.
No Qt event loop timing dependence.
"""

import os
import tempfile
import pytest
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.log_follower import LogFollower

def test_log_follower_pure_tail_logic():
    """Verify LogFollower correctly tails files and manages offsets without a timer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")
        
        follower = LogFollower(run_dir=tmpdir, poll_ms=10)
        
        received = []
        follower.line_received.connect(received.append)

        # 1. Initial poll with no files - should not crash, nothing received
        follower._poll()
        assert len(received) == 0
        
        # 2. Add some stdout
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("line 1\nline 2\n")
        
        follower._poll()
        assert "OUT: line 1" in received
        assert "OUT: line 2" in received
        assert len(received) == 2
        
        # 3. Add more stdout and some stderr
        received.clear()
        with open(stdout_path, "a", encoding="utf-8") as f:
            f.write("line 3\n")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write("err 1\n")
            
        follower._poll()
        assert "OUT: line 3" in received
        assert "ERR: err 1" in received
        assert len(received) == 2
        
        # 4. Empty poll - nothing new
        received.clear()
        follower._poll()
        assert len(received) == 0

def test_log_follower_partial_write_utf8():
    """Verify partial UTF-8 sequences don't crash and are replaced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        received = []
        follower.line_received.connect(received.append)
        
        # Write valid + partial UTF-8
        # \xe2\x80\xa6 is ellipsis (...)
        # We write only \xe2\x80 (incomplete)
        with open(stdout_path, "wb") as f:
            f.write(b"Hello\xe2\x80\n")
            
        follower._poll()
        assert len(received) == 1
        # Should have replacement char for \xe2\x80
        assert "Hello" in received[0]
        assert "\ufffd" in received[0]

def test_log_follower_offset_reset_on_shrink():
    """Verify offset resets to 0 if file shrinks (e.g. truncated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        with open(stdout_path, "w") as f:
            f.write("Initial long content\n")
        
        follower._poll()
        assert follower._stdout_offset > 0
        
        # Truncate file
        with open(stdout_path, "w") as f:
            f.write("Short\n")
            
        received = []
        follower.line_received.connect(received.append)
        
        follower._poll()
        assert follower._stdout_offset == os.path.getsize(stdout_path)
        assert "OUT: Short" in received

def test_log_follower_partial_line_reconstruction():
    """Verify that lines split across polls are correctly reconstructed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        received = []
        follower.line_received.connect(received.append)
        
        # 1. Write fragment to stdout
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("PART")
        follower._poll()
        assert len(received) == 0
        assert follower._stdout_carry == "PART"
        
        # 2. Complete the line
        with open(stdout_path, "a", encoding="utf-8") as f:
            f.write("IAL\n")
        follower._poll()
        assert len(received) == 1
        assert received[0] == "OUT: PARTIAL"
        assert follower._stdout_carry == ""
        
        # 3. Multiple lines + fragment in one write
        received.clear()
        with open(stdout_path, "a", encoding="utf-8") as f:
            f.write("line 1\nline 2\nFRAG")
        follower._poll()
        assert len(received) == 2
        assert received[0] == "OUT: line 1"
        assert received[1] == "OUT: line 2"
        assert follower._stdout_carry == "FRAG"
        
        # 4. stderr fragment
        received.clear()
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write("ERRP")
        follower._poll()
        assert len(received) == 0
        assert follower._stderr_carry == "ERRP"
        
        with open(stderr_path, "a", encoding="utf-8") as f:
            f.write("ART\n")
        follower._poll()
        assert len(received) == 1
        assert received[0] == "ERR: ERRPART"
        assert follower._stderr_carry == ""

def test_log_follower_stop_flushes_carry():
    """Verify that stop() flushes any remaining unterminated carry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        received = []
        follower.line_received.connect(received.append)
        
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("FINAL")
        
        follower._poll()
        assert len(received) == 0
        
        follower.stop()
        assert len(received) == 1
        assert received[0] == "OUT: FINAL"

def test_log_follower_consecutive_duplicate_suppression():
    """Verify that exact consecutive duplicates are suppressed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        received = []
        follower.line_received.connect(received.append)
        
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("DUPE\nDUPE\nUNIQUE\n")
        
        follower._poll()
        assert len(received) == 2
        assert received[0] == "OUT: DUPE"
        assert received[1] == "OUT: UNIQUE"

def test_log_follower_non_consecutive_duplicates_preserved():
    """Verify that non-consecutive duplicates are NOT suppressed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        follower = LogFollower(run_dir=tmpdir)
        
        received = []
        follower.line_received.connect(received.append)
        
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("A\nB\nA\n")
        
        follower._poll()
        assert len(received) == 3
        assert received[0] == "OUT: A"
        assert received[1] == "OUT: B"
        assert received[2] == "OUT: A"
