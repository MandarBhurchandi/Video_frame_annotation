# Video Annotation Tool

A tool for annotating video files with timestamps and labels, particularly useful for marking UI transitions, scrolling, typing, and other user interface interactions.

## Features

- Frame-by-frame video navigation
- Play/Pause/Stop functionality
- Video timeline slider
- Multiple annotation classes (transition, scroll, typing, UI interaction)
- Export annotations to JSON format
- Real-time frame information display
- Support for various video formats

## Installation

### Local Setup

1. Clone the repository:
```bash
    git clone https://github.com/MandarBhurchandi/Video_frame_annotation.git
```
Install requirements:

```bash
    install -r requirements.txt
```

Run the application:

```bash
    python annotator.py
```
Docker Setup

Build and run using Docker Compose:

```bash 
    docker-compose up --build
```