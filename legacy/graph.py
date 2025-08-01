#!/usr/bin/env python3
import curses
import time
import csv

def read_data(filename):
    """Reads CSV data and returns a list of (episode, score) tuples."""
    data = []
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header if present.
            for row in reader:
                try:
                    episode = row[0]
                    # Try to convert episode to an int, but if not possible, keep as string.
                    try:
                        episode_val = int(episode)
                    except ValueError:
                        episode_val = episode
                    score = float(row[1])
                    data.append((episode_val, score))
                except (IndexError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return data

def draw_line(stdscr, x0, y0, x1, y1):
    """Draws a line between (x0, y0) and (x1, y1) using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        try:
            stdscr.addch(y0, x0, '*')
        except curses.error:
            pass  # Ignore if out-of-bound.
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def draw_axes(stdscr, left_margin, effective_height, graph_width):
    # Draw y-axis (vertical line) at the left margin.
    for y in range(effective_height + 1):
        try:
            stdscr.addch(y, left_margin, '|')
        except curses.error:
            pass

    # Draw x-axis (horizontal line) at the bottom of the plot area.
    x_axis_y = effective_height
    for x in range(left_margin, left_margin + graph_width):
        try:
            stdscr.addch(x_axis_y, x, '-')
        except curses.error:
            pass

def draw_x_labels(stdscr, points, sampled_data, effective_height, left_margin, graph_width):
    # We'll place the x-axis labels on the row below the x-axis.
    label_y = effective_height + 1
    n_points = len(sampled_data)
    if n_points == 0:
        return

    # Determine tick positions: choose up to 5 ticks (first, quarter, middle, three-quarters, last)
    num_ticks = 5 if n_points >= 5 else n_points
    tick_indices = [int(i * (n_points - 1) / (num_ticks - 1)) for i in range(num_ticks)] if num_ticks > 1 else [0]
    for idx in tick_indices:
        x, _ = points[idx]
        episode_label = str(sampled_data[idx][0])
        # To avoid overlap, we only print the label if it fits in the remaining width.
        try:
            stdscr.addstr(label_y, x, episode_label)
        except curses.error:
            pass

def draw_line_plot(stdscr, data):
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    # Set margins:
    left_margin = 6     # Reserve columns for y-axis labels.
    bottom_margin = 3   # One row for the x-axis line, one for x-axis labels, one for info.
    graph_width = w - left_margin
    graph_height = h - bottom_margin

    if not data:
        stdscr.addstr(0, 0, "No data available yet...")
        stdscr.refresh()
        return

    # Separate episodes and scores.
    episodes, scores = zip(*data)
    max_score = max(scores)

    # Use plot area excluding the x-axis line.
    effective_height = graph_height - 1
    scale = effective_height / max_score if max_score > 0 else 1

    # If there are more data points than available columns, sample the data.
    n_points = len(scores)
    if n_points > graph_width:
        step = n_points / graph_width
        sampled_data = [data[int(i * step)] for i in range(graph_width)]
    else:
        sampled_data = list(data)

    # Compute (x, y) positions for each sampled data point.
    points = []
    for i, (ep, score) in enumerate(sampled_data):
        x = left_margin + i
        y = int(effective_height - (score * scale))
        points.append((x, y))

    # Draw axes.
    draw_axes(stdscr, left_margin, effective_height, graph_width)

    # Draw the line plot connecting the points.
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        draw_line(stdscr, x0, y0, x1, y1)

    # Add y-axis labels: max at the top and 0 at the bottom.
    try:
        stdscr.addstr(0, 0, f"{max_score:.1f}")
        stdscr.addstr(effective_height, 0, "0")
    except curses.error:
        pass

    # Draw x-axis labels (episodes) below the x-axis.
    draw_x_labels(stdscr, points, sampled_data, effective_height, left_margin, graph_width)

    # Display additional info at the very bottom.
    info_line = h - 1
    try:
        stdscr.addstr(info_line, 0, f"Max Score: {max_score} | Press Ctrl+C to exit")
    except curses.error:
        pass

    stdscr.refresh()

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor.
    filename = 'score_history.csv'
    while True:
        data = read_data(filename)
        draw_line_plot(stdscr, data)
        time.sleep(1)  # Update every second.

if __name__ == '__main__':
    curses.wrapper(main)

