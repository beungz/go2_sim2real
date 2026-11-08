"""Overlay a command timeline on a recorded video.

Adds a top band with the current command in large text, a timeline bar coloured by command with
the switch times marked, a moving arrow at the current time, and optional push markers.

    python annotate_video.py in.mp4 out.mp4 --segments 0:stand,6:sit,14:stand --pushes 4,8,12,16

Defaults match scripts/record_push_video.py.
"""

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
COLORS = {"stand": (42, 120, 214), "sit": (235, 104, 52)}
WHITE, INK, MUTED = (255, 255, 255), (16, 16, 16), (205, 205, 205)
FALL = (200, 40, 40)
FALL_FLASH_S = 1.5

# layout for 1920x1080, scaled for other sizes
BAND_H = 232
MARGIN = 60
TITLE_Y, TITLE_SIZE = 14, 66
SWITCH_LABEL_Y, SWITCH_LABEL_SIZE = 100, 28
BAR_Y0, BAR_Y1 = 150, 196
SEG_LABEL_SIZE = 34
PUSH_Y, PUSH_SIZE = 202, 20
ARROW_W, ARROW_H = 30, 20


def parse_segments(text: str) -> list[tuple[float, str]]:
    """Parse ``start_s:command,...`` into a time-sorted list of ``(start_s, command)``."""
    segs = []
    for item in text.split(","):
        t, cmd = item.split(":")
        segs.append((float(t), cmd.strip().lower()))
    return sorted(segs)


def command_at(segs, t: float) -> str:
    """Return the command active at time ``t``."""
    current = segs[0][1]
    for start, cmd in segs:
        if t >= start:
            current = cmd
    return current


def main():
    """Draw the timeline, pushes, falls and counters on every frame and write the new video."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--segments", default="0:stand,6:sit,14:stand", help="start_s:command,... (stand or sit)")
    p.add_argument("--pushes", default="4,8,12,16", help="push times in s; empty for none")
    p.add_argument("--falls", default="", help="fall times in s")
    p.add_argument("--events", help="events.json from record_push_video.py; overrides the three lists above")
    p.add_argument("--title", default="", help="caption shown bottom-left, e.g. the policy version")
    args = p.parse_args()

    segs = parse_segments(args.segments)
    pushes = [float(x) for x in args.pushes.split(",") if x.strip()]
    falls = [float(x) for x in args.falls.split(",") if x.strip()]
    if args.events:
        with open(args.events) as f:
            ev = json.load(f)
        segs = sorted((float(t), c) for t, c in ev["segments"])
        pushes, falls = ev["pushes"], ev["falls"]
    num_envs = ev.get("num_envs", 1) if args.events else 1
    off_pose = ev.get("off_pose") if args.events else None
    reader = imageio.get_reader(args.src)
    meta = reader.get_meta_data()
    fps = meta["fps"]
    n_frames = reader.count_frames()
    duration = n_frames / fps
    w, h = meta["size"]
    s = h / 1080.0  # scale factor for the layout constants

    def font(size):
        """Load the overlay font at ``size``, scaled to the video height."""
        return ImageFont.truetype(FONT, max(10, round(size * s)))

    def x_at(t):
        """Map time ``t`` in seconds to an x pixel on the timeline bar."""
        return MARGIN * s + (w - 2 * MARGIN * s) * min(max(t / duration, 0.0), 1.0)

    static = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(static)
    d.rectangle([0, 0, w, BAND_H * s], fill=(12, 12, 12, 185))
    bounds = [start for start, _ in segs] + [duration]
    y0, y1 = BAR_Y0 * s, BAR_Y1 * s
    for (start, cmd), end in zip(segs, bounds[1:]):
        xa, xb = x_at(start), x_at(end)
        d.rectangle([xa, y0, xb, y1], fill=COLORS[cmd] + (255,))
        label = cmd.upper()
        f = font(SEG_LABEL_SIZE)
        tw, th = d.textbbox((0, 0), label, font=f)[2:]
        if tw < xb - xa - 10 * s:
            d.text(((xa + xb - tw) / 2, (y0 + y1) / 2), label, font=f, fill=WHITE, anchor="lm")
    f = font(SWITCH_LABEL_SIZE)
    for i, t in enumerate(bounds):
        x = x_at(t)
        if 0 < i < len(bounds) - 1:
            d.line([x, SWITCH_LABEL_Y * s + 34 * s, x, y1], fill=WHITE, width=max(2, round(4 * s)))
        anchor = "la" if i == 0 else "ra" if i == len(bounds) - 1 else "ma"
        d.text((x, SWITCH_LABEL_Y * s), f"{t:g} s", font=f, fill=WHITE, anchor=anchor)
    f = font(PUSH_SIZE)
    for t in pushes:
        x = x_at(t)
        d.polygon([(x, y1 + 2 * s), (x - 7 * s, y1 + 12 * s), (x + 7 * s, y1 + 12 * s)], fill=MUTED + (255,))
        d.text((x + 10 * s, PUSH_Y * s), "push", font=f, fill=MUTED, anchor="la")
    for t in falls:
        x, yc, r = x_at(t), (y0 + y1) / 2, 15 * s
        for col, wd in ((INK, 10), (FALL, 5)):
            d.line([x - r, yc - r, x + r, yc + r], fill=col, width=max(2, round(wd * s)))
            d.line([x - r, yc + r, x + r, yc - r], fill=col, width=max(2, round(wd * s)))
    if args.title:
        f = font(TITLE_SIZE * 0.6)
        box = d.textbbox((MARGIN * s, h - 40 * s), args.title, font=f, anchor="lb")
        d.rounded_rectangle([box[0] - 16 * s, box[1] - 12 * s, box[2] + 16 * s, box[3] + 12 * s],
                            radius=10 * s, fill=(12, 12, 12, 200))
        d.text((MARGIN * s, h - 40 * s), args.title, font=f, fill=WHITE, anchor="lb")

    writer = imageio.get_writer(args.dst, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=8)
    title_font, time_font, banner_font = font(TITLE_SIZE), font(TITLE_SIZE * 0.62), font(TITLE_SIZE * 0.8)
    for k, frame in enumerate(reader):
        t = k / fps
        img = Image.fromarray(frame).convert("RGBA")
        img.alpha_composite(static)
        d = ImageDraw.Draw(img)
        cmd = command_at(segs, t)
        d.text((MARGIN * s, TITLE_Y * s), "COMMAND:", font=title_font, fill=WHITE)
        cx = MARGIN * s + d.textlength("COMMAND: ", font=title_font)
        tw = d.textlength(cmd.upper(), font=title_font)
        d.rounded_rectangle([cx - 12 * s, TITLE_Y * s - 2 * s, cx + tw + 12 * s, TITLE_Y * s + 80 * s],
                            radius=10 * s, fill=COLORS[cmd] + (255,))
        d.text((cx, TITLE_Y * s), cmd.upper(), font=title_font, fill=WHITE)
        d.text((w - MARGIN * s, TITLE_Y * s + 14 * s), f"t = {t:4.1f} s", font=time_font, fill=WHITE, anchor="ra")
        x = x_at(t)
        d.line([x, y0 - 2 * s, x, y1 + 2 * s], fill=INK, width=max(3, round(8 * s)))
        d.line([x, y0 - 2 * s, x, y1 + 2 * s], fill=WHITE, width=max(2, round(4 * s)))
        tip = y0 - 3 * s
        tri = [(x, tip), (x - ARROW_W / 2 * s, tip - ARROW_H * s), (x + ARROW_W / 2 * s, tip - ARROW_H * s)]
        d.polygon(tri, fill=WHITE, outline=INK, width=max(1, round(2 * s)))
        recent = sum(0 <= t - tf < FALL_FLASH_S for tf in falls)
        if recent:
            if num_envs == 1:
                label = "✕  FELL – robot reset"
            else:
                label = f"✕  {recent} robot{'s' if recent > 1 else ''} fell – reset"
            box = d.textbbox((w / 2, (BAND_H + 30) * s), label, font=banner_font, anchor="mt")
            d.rounded_rectangle([box[0] - 24 * s, box[1] - 14 * s, box[2] + 24 * s, box[3] + 14 * s],
                                radius=12 * s, fill=FALL + (235,))
            d.text((w / 2, (BAND_H + 30) * s), label, font=banner_font, fill=WHITE, anchor="mt")
        if num_envs > 1:
            fell = sum(tf <= t for tf in falls)
            lines = [f"{num_envs} robots", f"Fell so far: {fell}"]
            if off_pose:
                lines.append(f"Off pose now: {off_pose[min(k, len(off_pose) - 1)]}")
            f = font(TITLE_SIZE * 0.6)
            gap = 58 * s
            x_r, y_b = w - MARGIN * s, h - 40 * s
            top = y_b - gap * len(lines)
            width = max(d.textlength(line, font=f) for line in lines)
            d.rounded_rectangle([x_r - width - 20 * s, top - 6 * s, x_r + 20 * s, y_b + 12 * s],
                                radius=10 * s, fill=(12, 12, 12, 200))
            for i, line in enumerate(lines):
                col = (255, 120, 120) if line.startswith("Fell") and fell else WHITE
                d.text((x_r, top + gap * (i + 1)), line, font=f, fill=col, anchor="rb")
        writer.append_data(np.asarray(img.convert("RGB")))
    writer.close()
    print(f"wrote {args.dst}: {n_frames} frames, {duration:.1f} s, {w}x{h}")


if __name__ == "__main__":
    main()
