import gradio as gr
import json
from datetime import datetime
import cv2
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union, Dict
import pandas as pd
import time
import threading

@dataclass
class Annotation:
    class_id: str
    start_time: float
    end_time: float
    duration: float

@dataclass
class VideoInfo:
    file_path: str
    fps: float
    duration: float
    frame_count: int
    width: int
    height: int
    timestamp: str

class VideoAnnotator:
    def __init__(self):
        self.annotations = []
        self.current_video_info = None
        self.current_start_time = None
        self.cap = None
        self.current_frame_number = 0
        self.is_playing = False
        self.lock = threading.Lock()  

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame by number"""
        if self.cap is None or frame_number < 0:
            return None
            
        if frame_number >= self.current_video_info.frame_count:
            frame_number = self.current_video_info.frame_count - 1
        
        with self.lock:  # Add lock around video operations
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_number = frame_number
            return frame
        return None


    def load_video(self, video_path: str) -> Tuple[str, np.ndarray, gr.Slider]:
        """Load video and return its information"""
        try:
            if self.cap is not None:
                self.is_playing = False
                self.cap.release()
                
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                return "Error: Could not open video file", None, gr.Slider()
            
            self.current_video_info = VideoInfo(
                file_path=video_path,
                fps=self.cap.get(cv2.CAP_PROP_FPS),
                duration=self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS),
                frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                timestamp=datetime.now().isoformat()
            )
            
            self.current_frame_number = 0
            self.annotations = []
            
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return (
                    f"Video loaded successfully:\nDuration: {self.current_video_info.duration:.2f}s\n"
                    f"FPS: {self.current_video_info.fps}\nTotal frames: {self.current_video_info.frame_count}",
                    frame,
                    gr.Slider(minimum=0, maximum=self.current_video_info.duration, 
                             step=1/self.current_video_info.fps, value=0)
                )
            return "Error: Could not read first frame", None, gr.Slider()
            
        except Exception as e:
            return f"Error loading video: {str(e)}", None, gr.Slider()

    def update_frame_from_slider(self, time_point: Union[float, Dict]) -> Tuple[Optional[np.ndarray], str]:
        """Update frame based on slider position"""
        if isinstance(time_point, dict):
            time_point = float(time_point.get('value', 0))
        else:
            time_point = float(time_point)
            
        frame_number = int(time_point * self.current_video_info.fps)
        frame = self.get_frame(frame_number)
        
        if frame is not None:
            return frame, f"Frame: {frame_number} / Time: {time_point:.2f}s"
        return None, "Failed to read frame"

    def next_frame(self) -> Tuple[Optional[np.ndarray], float, str]:
        """Move to next frame"""
        if self.current_frame_number >= self.current_video_info.frame_count - 1:
            self.is_playing = False
            return None, self.current_frame_number / self.current_video_info.fps, "End of video"
        
        frame = self.get_frame(self.current_frame_number + 1)
        time_point = self.current_frame_number / self.current_video_info.fps
        
        if frame is not None:
            return frame, time_point, f"Frame: {self.current_frame_number} / Time: {time_point:.2f}s"
        return None, time_point, "Failed to read frame"

    def prev_frame(self) -> Tuple[Optional[np.ndarray], float, str]:
        """Move to previous frame"""
        if self.current_frame_number <= 0:
            return None, 0, "Start of video"
            
        frame = self.get_frame(self.current_frame_number - 1)
        time_point = self.current_frame_number / self.current_video_info.fps
        
        if frame is not None:
            return frame, time_point, f"Frame: {self.current_frame_number} / Time: {time_point:.2f}s"
        return None, time_point, "Failed to read frame"

    def play_pause(self) -> Tuple[Optional[np.ndarray], float, str, str]:
        """Toggle play/pause state"""
        if self.cap is None or self.current_video_info is None:
            return None, 0, "No video loaded", "Play ▶"
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            return (
                self.get_frame(self.current_frame_number),
                self.current_frame_number / self.current_video_info.fps,
                f"Frame: {self.current_frame_number} / Time: {self.current_frame_number / self.current_video_info.fps:.2f}s",
                "Pause ⏸"
            )
        else:
            return (
                self.get_frame(self.current_frame_number),
                self.current_frame_number / self.current_video_info.fps,
                f"Frame: {self.current_frame_number} / Time: {self.current_frame_number / self.current_video_info.fps:.2f}s",
                "Play ▶"
            )

    def stop(self) -> Tuple[Optional[np.ndarray], float, str]:
        """Stop playback and reset to start"""
        frame, time_value, info, btn_text = annotator.play_pause()
        self.current_frame_number = 0
        frame = self.get_frame(0)
        
        if frame is not None:
            return frame, 0, "Video stopped", btn_text
        return None, 0, "Failed to read frame", "Play ▶"

    def mark_start(self) -> str:
        """Mark the start time of an annotation"""
        if self.current_video_info is None:
            return "Please load a video first"
        
        self.current_start_time = self.current_frame_number / self.current_video_info.fps
        return f"Start time marked at {self.current_start_time:.2f}s (Frame {self.current_frame_number})"

    def mark_end(self, class_id: str) -> Tuple[pd.DataFrame, str]:
        """Mark the end time and create an annotation"""
        if self.current_video_info is None:
            return pd.DataFrame(), "Please load a video first"
        
        if self.current_start_time is None:
            return pd.DataFrame(), "Please mark start time first"
        
        end_time = self.current_frame_number / self.current_video_info.fps
        if end_time <= self.current_start_time:
            return pd.DataFrame(), "End time must be after start time"
        
        annotation = Annotation(
            class_id=class_id,
            start_time=self.current_start_time,
            end_time=end_time,
            duration=end_time - self.current_start_time
        )
        
        self.annotations.append(annotation)
        self.current_start_time = None
        
        df = pd.DataFrame([asdict(a) for a in self.annotations])
        return df, f"Annotation added: {class_id} from {annotation.start_time:.2f}s to {end_time:.2f}s"

    def export_annotations(self, output_path: str) -> str:
        """Export annotations to JSON"""
        if self.current_video_info is None:
            return "Please load a video first"
        
        if not self.annotations:
            return "No annotations to export"
        
        output = {
            "video_info": asdict(self.current_video_info),
            "annotations": [asdict(a) for a in self.annotations]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            return f"Annotations exported to {output_path}"
        except Exception as e:
            return f"Error exporting annotations: {str(e)}"

    def clear_annotations(self) -> pd.DataFrame:
        """Clear all annotations"""
        self.annotations = []
        self.current_start_time = None
        return pd.DataFrame()
        
    def __del__(self):
        if self.cap is not None:
            with self.lock:
                self.cap.release()


def create_ui():
    annotator = VideoAnnotator()
    
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                video_path = gr.Text(label="Video Path")
                load_btn = gr.Button("Load Video")
                video_output = video_output = gr.Image(
                                    label="Video Frame",
                                    width=300
                                )
                
                # Video controls
                with gr.Row():
                    play_pause_btn = gr.Button("Play ▶")
                    stop_btn = gr.Button("Stop ⏹")
                
                # Video slider
                time_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=0.001,
                    label="Video Progress",
                    visible=False
                )
                
                # Frame navigation
                with gr.Row():
                    prev_frame_btn = gr.Button("◀ Previous Frame")
                    next_frame_btn = gr.Button("Next Frame ▶")
                
                frame_info = gr.Text(label="Current Frame", value="Frame: 0 / Time: 0.00s")

            with gr.Column():
                status_text = gr.Text(label="Status")
                class_input = gr.Dropdown(
                    choices=["transition", "scroll", "typing", "ui_interaction"],
                    value="transition",
                    label="Annotation Class"
                )
                
                with gr.Row():
                    mark_start_btn = gr.Button("Mark Start")
                    mark_end_btn = gr.Button("Mark End")
                    clear_btn = gr.Button("Clear Annotations")
                
                annotations_display = gr.DataFrame(
                    headers=["class_id", "start_time", "end_time", "duration"],
                    label="Annotations"
                )
                
                export_path = gr.Text(label="Export Path", placeholder="annotations.json")
                export_btn = gr.Button("Export Annotations")

        def play_update():
            if annotator.is_playing:
                try:
                    frame, time_value, info = annotator.next_frame()
                    if frame is not None:
                        return frame, time_value, info
                    else:
                        # Stop at end of video
                        annotator.is_playing = False
                except Exception as e:
                    print(f"Error in play_update: {e}")
                    annotator.is_playing = False
            return gr.skip(), gr.skip(), gr.skip()  # Skip update if not playing


        # Use a slower timer rate to prevent memory issues
        timer = gr.Timer(1/60)  # 10 FPS instead of 30
        timer.tick(
            fn=play_update,
            outputs=[video_output, time_slider, frame_info]
        )

        def toggle_play():
            frame, time_value, info, btn_text = annotator.play_pause()
            return frame, time_value, info, btn_text, gr.Timer(1/10, active=annotator.is_playing)


        # Make sure to stop playback when stopping
        def safe_stop():
            annotator.is_playing = False
            return annotator.stop()

        play_pause_btn.click(
            fn=toggle_play,
            outputs=[video_output, time_slider, frame_info, play_pause_btn, timer]
        )
        # Event handlers
        load_btn.click(
            fn=annotator.load_video,
            inputs=[video_path],
            outputs=[status_text, video_output, time_slider]
        )
        
        stop_btn.click(
            fn=annotator.stop,
            outputs=[video_output, time_slider, play_pause_btn]
        )
        
        time_slider.release(
            fn=annotator.update_frame_from_slider,
            inputs=[time_slider],
            outputs=[video_output, frame_info]
        )
        
        next_frame_btn.click(
            fn=annotator.next_frame,
            outputs=[video_output, time_slider, frame_info]
        )
        
        prev_frame_btn.click(
            fn=annotator.prev_frame,
            outputs=[video_output, time_slider, frame_info]
        )
        
        mark_start_btn.click(
            fn=annotator.mark_start,
            outputs=[status_text]
        )
        
        mark_end_btn.click(
            fn=annotator.mark_end,
            inputs=[class_input],
            outputs=[annotations_display, status_text]
        )
        
        clear_btn.click(
            fn=annotator.clear_annotations,
            outputs=[annotations_display]
        )
        
        export_btn.click(
            fn=annotator.export_annotations,
            inputs=[export_path],
            outputs=[status_text]
        )
    return app

if __name__ == "__main__":
    app = create_ui()
    app.queue()  # Enable queue for background tasks
    app.launch(share=True)