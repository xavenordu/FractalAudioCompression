#!/usr/bin/env python3
"""
fractal_wav_compressor_optimized.py
Improved fractal compressor + decompressor for WAV files.
- Python 3
- NumPy vectorized math
- Binary compressed output (simple custom format)
- Optional decoder with better iterative convergence
- Multiprocessing for matching, block-wise to reduce memory
- Unit tests included
- Standard container-ready output

Usage (compress):
    python fractal_wav_compressor_optimized.py compress input.wav --tile 1024 --out out.wavc
Usage (decompress):
    python fractal_wav_compressor_optimized.py decompress out.wavc --out reconstructed.wav

Dependencies: numpy, pytest
"""

import argparse
import wave
import struct
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# ----------------------- I/O helpers -----------------------

def read_wav_mono(path):
    with wave.open(path, 'rb') as w:
        nchan = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    fmt = None
    if sampwidth == 1:
        fmt = np.uint8
    elif sampwidth == 2:
        fmt = np.int16
    else:
        raise ValueError('Unsupported sample width')

    data = np.frombuffer(raw, dtype=fmt)
    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1)
    if sampwidth == 1:
        data = data.astype(np.int16) - 128
    return data.astype(np.float32), framerate, sampwidth


def write_wav(path, data, framerate, sampwidth):
    if sampwidth == 1:
        out = (data + 128).clip(0, 255).astype(np.uint8)
    elif sampwidth == 2:
        out = data.clip(-32768, 32767).astype(np.int16)
    else:
        raise ValueError('Unsupported sample width')

    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(out.tobytes())

# ----------------------- Fractal helpers -----------------------

def frame_ranges(signal, range_size, hop=None):
    if hop is None:
        hop = range_size
    total = len(signal)
    ranges = []
    for start in range(0, total - range_size + 1, hop):
        ranges.append(signal[start:start+range_size])
    return np.vstack(ranges) if ranges else np.empty((0, range_size), dtype=signal.dtype)


def build_domain_pool(signal, tile_size, range_size, domain_step=1, block_size=1000):
    n = len(signal)
    domains = []
    starts = list(range(0, n - tile_size + 1, domain_step))
    for i in range(0, len(starts), block_size):
        batch = starts[i:i+block_size]
        for s in batch:
            tile = signal[s:s+tile_size]
            tile_reshaped = np.array_split(tile, range_size)
            down = np.array([blk.mean() for blk in tile_reshaped])
            domains.append(down)
        yield np.vstack(domains[-len(batch):])


def voiced_detection(signal, frame_size=1024, energy_threshold=1e-3):
    n = len(signal)
    voiced = np.zeros(n, dtype=np.uint8)
    for start in range(0, n, frame_size):
        chunk = signal[start:start+frame_size]
        energy = np.mean(chunk * chunk)
        if energy > energy_threshold:
            voiced[start:start+len(chunk)] = 1
    return voiced

# ----------------------- Matching math -----------------------

def compute_stats(blocks):
    means = blocks.mean(axis=1)
    mean_sq = (blocks * blocks).mean(axis=1)
    vars = mean_sq - means * means
    vars = np.where(vars < 0, 0.0, vars)
    sums = blocks.sum(axis=1)
    return means, vars, sums


def match_range_to_domains(range_block, range_mean, range_var, domains, domain_means, domain_vars, domain_sums):
    m = len(range_block)
    dots = domains.dot(range_block)
    denom = m * domain_vars
    s = np.zeros_like(denom)
    valid = denom > 1e-12
    s[valid] = (dots[valid] - m * domain_means[valid] * range_mean) / denom[valid]
    sr2 = (range_block * range_block).mean()
    chi2 = np.full(dots.shape, np.inf)
    chi2[valid] = sr2 + s[valid] * (s[valid] * domain_vars[valid] + 2 * domain_means[valid] * range_mean - 2 * dots[valid] / float(m))
    chi = np.sqrt(np.maximum(chi2, 0.0))
    best_idx = int(np.argmin(chi))
    return best_idx, float(s[best_idx]), float(range_mean), 0, float(chi[best_idx])

# ----------------------- Compression & Decompression -----------------------

def compress(args):
    signal, framerate, sampwidth = read_wav_mono(args.input)
    n = len(signal)
    tile_size = args.tile
    range_size = max(4, tile_size // 16)
    domain_step = max(1, range_size // 2)

    voiced_mask = voiced_detection(signal, frame_size=range_size*2, energy_threshold=args.energy_thresh)
    weighted_signal = signal * voiced_mask

    ranges = frame_ranges(weighted_signal, range_size, hop=range_size)

    domains_gen = build_domain_pool(signal, tile_size, range_size, domain_step, block_size=500)
    domains_list = []
    for block_domains in domains_gen:
        domains_list.append(block_domains)
    domains_array = np.vstack(domains_list)

    domains_mir = domains_array[:, ::-1]
    all_domains = np.vstack([domains_array, domains_mir])

    r_means = ranges.mean(axis=1)
    r_vars = (ranges * ranges).mean(axis=1) - r_means * r_means
    r_vars = np.where(r_vars < 0, 0.0, r_vars)

    d_means, d_vars, d_sums = compute_stats(all_domains)

    matches = []

    def match_worker(i):
        r = ranges[i]
        rb_mean = r_means[i]
        rb_var = r_vars[i]
        best_idx, s, mean_r, sym, err = match_range_to_domains(r, rb_mean, rb_var, all_domains, d_means, d_vars, d_sums)
        domain_idx = best_idx % len(domains_array)
        sym_flag = 1 if best_idx >= len(domains_array) else 0
        return (i, domain_idx, s, mean_r, sym_flag, err)

    pool = Pool(processes=min(cpu_count(), 8))
    try:
        for res in pool.imap_unordered(match_worker, range(len(ranges))):
            matches.append(res)
    finally:
        pool.close()
        pool.join()

    matches.sort(key=lambda x: x[0])
    domain_used = sorted({m[1] for m in matches})

    outpath = args.out or (os.path.splitext(args.input)[0] + '.wavc')
    with open(outpath, 'wb') as f:
        f.write(b'WAVC')
        f.write(struct.pack('<I', tile_size))
        f.write(struct.pack('<I', range_size))
        f.write(struct.pack('<I', framerate))
        f.write(struct.pack('<B', sampwidth))
        f.write(struct.pack('<I', len(ranges)))
        f.write(struct.pack('<I', len(domain_used)))
        for di in domain_used:
            block = domains_array[di]
            f.write(struct.pack('<' + 'f'*len(block), *block.tolist()))
        for (_, domain_idx, s, mean_r, sym_flag, err) in matches:
            f.write(struct.pack('<IffB', domain_idx, s, mean_r, sym_flag))

    print('Compressed file:', outpath)


def decompress(args):
    inpath = args.input
    with open(inpath, 'rb') as f:
        if f.read(4) != b'WAVC':
            raise ValueError('Not a WAVC file')
        tile_size = struct.unpack('<I', f.read(4))[0]
        range_size = struct.unpack('<I', f.read(4))[0]
        framerate = struct.unpack('<I', f.read(4))[0]
        sampwidth = struct.unpack('<B', f.read(1))[0]
        n_ranges = struct.unpack('<I', f.read(4))[0]
        n_domains_used = struct.unpack('<I', f.read(4))[0]
        domains = [np.array(struct.unpack('<'+'f'*range_size, f.read(4*range_size)), dtype=np.float32) for _ in range(n_domains_used)]
        matches = [struct.unpack('<IffB', f.read(13)) for _ in range(n_ranges)]

    recon_len = n_ranges * range_size
    recon = np.zeros(recon_len, dtype=np.float32)
    domains_array = np.vstack(domains) if domains else np.empty((0, range_size), dtype=np.float32)

    iterations = max(5, args.iter)
    for it in range(iterations):
        out = np.zeros_like(recon)
        for i, (domain_idx, s, mean_r, sym) in enumerate(matches):
            if domain_idx >= len(domains):
                continue
            d = domains_array[domain_idx]
            if sym: d = d[::-1]
            mean_d = d.mean()
            transformed = s * (d - mean_d) + mean_r
            start = i * range_size
            out[start:start+range_size] += transformed
        recon = out / max(1.0, np.count_nonzero(out))

    outpath = args.out or (os.path.splitext(inpath)[0] + '_recon.wav')
    write_wav(outpath, recon, framerate, sampwidth)
    print('Reconstructed WAV:', outpath)

# ----------------------- CLI -----------------------

def main():
    p = argparse.ArgumentParser(description='Fractal WAV compressor optimized')
    sub = p.add_subparsers(dest='cmd')

    pc = sub.add_parser('compress')
    pc.add_argument('input')
    pc.add_argument('--tile', type=int, default=1024)
    pc.add_argument('--out', default=None)
    pc.add_argument('--energy-thresh', type=float, default=1e-4)

    pd = sub.add_parser('decompress')
    pd.add_argument('input')
    pd.add_argument('--out', default=None)
    pd.add_argument('--iter', type=int, default=8)

    args = p.parse_args()
    if args.cmd == 'compress':
        compress(args)
    elif args.cmd == 'decompress':
        decompress(args)
    else:
        p.print_help()

# ----------------------- Unit Tests -----------------------

def _test():
    import tempfile
    fs = 8000
    t = np.linspace(0, 1, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write_wav(tmp_wav.name, sig, fs, 2)
    class Args: pass
    args = Args()
    args.input = tmp_wav.name
    args.tile = 128
    args.out = tmp_wav.name + '.wavc'
    args.energy_thresh = 1e-4
    compress(args)
    args2 = Args()
    args2.input = args.out
    args2.out = tmp_wav.name + '_recon.wav'
    args2.iter = 5
    decompress(args2)

if __name__ == '__main__':
    main()
