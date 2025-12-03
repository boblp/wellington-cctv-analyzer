"""
Advanced Security-Focused Object Detection and Tracking using Ultralytics YOLO

This script provides comprehensive object tracking capabilities for security and surveillance
applications, utilizing all available Ultralytics tracking features including:
- Zone monitoring and loitering detection
- Line crossing detection
- Speed estimation and direction tracking
- Object counting and statistics
- Alert systems with optional email notifications
- Multi-threading support for multiple cameras
- Advanced tracking with ReID support
- CSV and JSON data export
- Real-time stream processing
- BEHAVIOR ANALYSIS: Suspicious activity detection (theft, violence, aggressive behavior)
- Pose estimation for violence detection
- Proximity analysis for altercation detection
- Anomaly detection for unusual patterns

Based on: https://docs.ultralytics.com/es/modes/track/
"""

from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import json
import csv
import time
import threading
from ultralytics import YOLO

# Optional: Import SecurityAlarm if available
try:
    from ultralytics.solutions import SecurityAlarm
    SECURITY_ALARM_AVAILABLE = True
except ImportError:
    SECURITY_ALARM_AVAILABLE = False
    print("Note: SecurityAlarm not available. Install ultralytics solutions for email alerts.")


class LineCrossing:
    """Represents a line for crossing detection."""
    
    def __init__(self, name: str, point1: Tuple[int, int], point2: Tuple[int, int], direction: str = "both"):
        """
        Initialize a crossing line.
        
        Args:
            name: Line identifier
            point1: First point (x, y)
            point2: Second point (x, y)
            direction: 'forward', 'backward', or 'both'
        """
        self.name = name
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.direction = direction
        self.crossings = {'forward': 0, 'backward': 0}
        self.last_side = {}  # track_id -> 'left' or 'right'
    
    def check_crossing(self, track_id: int, current_pos: Tuple[float, float], prev_pos: Optional[Tuple[float, float]] = None) -> Optional[str]:
        """Check if an object crossed the line. Returns 'forward', 'backward', or None."""
        if prev_pos is None:
            return None
        
        # Vector from point1 to point2
        line_vec = self.point2 - self.point1
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return None
        
        # Normalize line vector
        line_vec_norm = line_vec / line_length
        
        # Vector perpendicular to line (pointing to one side)
        perp_vec = np.array([-line_vec_norm[1], line_vec_norm[0]])
        
        # Check which side current and previous positions are on
        prev_vec = np.array(prev_pos) - self.point1
        curr_vec = np.array(current_pos) - self.point1
        
        prev_side = np.sign(np.dot(prev_vec, perp_vec))
        curr_side = np.sign(np.dot(curr_vec, perp_vec))
        
        # Check if crossed
        if prev_side != curr_side and prev_side != 0 and curr_side != 0:
            # Determine direction based on movement
            movement_vec = np.array(current_pos) - np.array(prev_pos)
            movement_along_line = np.dot(movement_vec, line_vec_norm)
            
            if movement_along_line > 0:
                crossing_dir = 'forward'
            else:
                crossing_dir = 'backward'
            
            # Check if direction is allowed
            if self.direction in ['both', crossing_dir]:
                self.crossings[crossing_dir] += 1
                self.last_side[track_id] = 'right' if curr_side > 0 else 'left'
                return crossing_dir
        
        self.last_side[track_id] = 'right' if curr_side > 0 else 'left'
        return None


class BehaviorAnalyzer:
    """Analyzes object behaviors to detect suspicious activities like theft and violence."""
    
    def __init__(
        self,
        violence_threshold: float = 0.2,
        proximity_threshold: float = 10.0,
        rapid_movement_threshold: float = 5.0,  # m/s
        sudden_stop_threshold: float = 0.6,  # m/s change
        erratic_movement_window: int = 10
    ):
        """
        Initialize behavior analyzer.
        
        Args:
            violence_threshold: Threshold for violence detection (0-1)
            proximity_threshold: Distance in pixels for proximity alerts
            rapid_movement_threshold: Speed threshold for rapid movement (m/s)
            sudden_stop_threshold: Speed change threshold for sudden stops
            erratic_movement_window: Frames to analyze for erratic movement
        """
        self.violence_threshold = violence_threshold
        self.proximity_threshold = proximity_threshold
        self.rapid_movement_threshold = rapid_movement_threshold
        self.sudden_stop_threshold = sudden_stop_threshold
        self.erratic_movement_window = erratic_movement_window
        
        # Behavior history
        self.proximity_alerts = deque(maxlen=100)
        self.rapid_movements = deque(maxlen=100)
        self.suspicious_behaviors = deque(maxlen=200)
        
    def analyze_pose_for_violence(self, keypoints: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Analyze pose keypoints for signs of violence/aggression.
        
        Args:
            keypoints: Pose keypoints array (if using pose model)
            
        Returns:
            Dict with violence indicators and confidence scores
        """
        if keypoints is None or len(keypoints) == 0:
            return {'violence_score': 0.0, 'aggressive_pose': False}
        
        violence_score = 0.0
        indicators = []
        
        # COCO pose keypoint indices (if using COCO format)
        # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
        # 11-12: hips, 13-14: knees, 15-16: ankles
        
        try:
            # Check for raised arms (potential punching/throwing)
            if len(keypoints) > 10:
                # Get wrist and shoulder positions
                left_wrist = keypoints[9] if len(keypoints) > 9 else None
                right_wrist = keypoints[10] if len(keypoints) > 10 else None
                left_shoulder = keypoints[5] if len(keypoints) > 5 else None
                right_shoulder = keypoints[6] if len(keypoints) > 6 else None
                
                if (left_wrist is not None and left_shoulder is not None and 
                    len(left_wrist) >= 2 and len(left_shoulder) >= 2):
                    # Check if wrist is above shoulder (raised arm)
                    if left_wrist[1] < left_shoulder[1] - 20:  # y-coordinate (top is smaller)
                        violence_score += 0.3
                        indicators.append('raised_left_arm')
                
                if (right_wrist is not None and right_shoulder is not None and
                    len(right_wrist) >= 2 and len(right_shoulder) >= 2):
                    if right_wrist[1] < right_shoulder[1] - 20:
                        violence_score += 0.3
                        indicators.append('raised_right_arm')
                
                # Check for wide stance (aggressive posture)
                if len(keypoints) > 12:
                    left_hip = keypoints[11] if len(keypoints) > 11 else None
                    right_hip = keypoints[12] if len(keypoints) > 12 else None
                    
                    if (left_hip is not None and right_hip is not None and
                        len(left_hip) >= 2 and len(right_hip) >= 2):
                        hip_distance = abs(left_hip[0] - right_hip[0])
                        if hip_distance > 50:  # Wide stance
                            violence_score += 0.2
                            indicators.append('wide_stance')
        except Exception as e:
            pass  # Continue even if pose analysis fails
        
        violence_score = min(violence_score, 1.0)
        
        return {
            'violence_score': violence_score,
            'aggressive_pose': violence_score >= self.violence_threshold,
            'indicators': indicators
        }
    
    def check_proximity(
        self,
        track_id1: int,
        pos1: Tuple[float, float],
        track_id2: int,
        pos2: Tuple[float, float],
        class1: str,
        class2: str
    ) -> Optional[Dict]:
        """
        Check if two objects are too close (potential altercation).
        
        Args:
            track_id1, track_id2: Track IDs
            pos1, pos2: Positions (x, y)
            class1, class2: Object classes
            
        Returns:
            Alert dict if proximity is suspicious, None otherwise
        """
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        
        # Only alert for person-to-person or person-to-object proximity
        if distance < self.proximity_threshold:
            # Person-to-person proximity (potential fight)
            if class1 == 'person' and class2 == 'person':
                return {
                    'type': 'suspicious_proximity',
                    'subtype': 'person_to_person',
                    'track_id1': int(track_id1),
                    'track_id2': int(track_id2),
                    'distance': float(distance),
                    'severity': 'high',
                    'description': 'Two people in close proximity - potential altercation'
                }
            # Person near valuable object (potential theft)
            elif class1 == 'person' and class2 in ['handbag', 'backpack', 'suitcase', 'laptop']:
                return {
                    'type': 'suspicious_proximity',
                    'subtype': 'person_near_object',
                    'track_id1': int(track_id1),
                    'track_id2': int(track_id2),
                    'distance': float(distance),
                    'severity': 'medium',
                    'description': f'Person near {class2} - potential theft'
                }
        
        return None
    
    def detect_rapid_movement(self, speed: float, track_id: int) -> Optional[Dict]:
        """Detect rapid movement (running, fleeing)."""
        if speed > self.rapid_movement_threshold:
            return {
                'type': 'rapid_movement',
                'track_id': int(track_id),
                'speed': float(speed),
                'severity': 'medium',
                'description': f'Rapid movement detected ({speed:.2f} m/s) - possible fleeing or running'
            }
        return None
    
    def detect_sudden_stop(self, speed_history: deque) -> Optional[Dict]:
        """Detect sudden stop (suspicious behavior)."""
        if len(speed_history) < 5:
            return None
        
        recent_speeds = list(speed_history)[-5:]
        avg_speed_before = np.mean(recent_speeds[:-1])
        current_speed = recent_speeds[-1]
        
        speed_change = avg_speed_before - current_speed
        
        if avg_speed_before > 2.0 and speed_change > self.sudden_stop_threshold:
            return {
                'type': 'sudden_stop',
                'speed_before': float(avg_speed_before),
                'speed_after': float(current_speed),
                'severity': 'low',
                'description': 'Sudden stop detected - unusual behavior pattern'
            }
        
        return None
    
    def detect_erratic_movement(self, track_history: deque) -> Optional[Dict]:
        """Detect erratic movement patterns."""
        if len(track_history) < self.erratic_movement_window:
            return None
        
        recent_points = list(track_history)[-self.erratic_movement_window:]
        points = np.array(recent_points)
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            angle = np.arctan2(dy, dx)
            directions.append(angle)
        
        # Calculate variance in directions (high variance = erratic)
        if len(directions) > 2:
            direction_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                # Normalize to 0-π
                change = min(change, 2 * np.pi - change)
                direction_changes.append(change)
            
            avg_change = np.mean(direction_changes)
            
            # High average change indicates erratic movement
            if avg_change > np.pi / 3:  # More than 60 degrees average change
                return {
                    'type': 'erratic_movement',
                    'avg_direction_change': float(avg_change),
                    'severity': 'medium',
                    'description': 'Erratic movement pattern detected - suspicious behavior'
                }
        
        return None
    
    def detect_theft_indicators(
        self,
        track_id: int,
        class_name: str,
        speed: float,
        zones_visited: set,
        loitering_time: int,
        near_valuable_objects: bool
    ) -> Optional[Dict]:
        """
        Detect potential theft indicators.
        
        Indicators:
        - Loitering near valuable items
        - Rapid exit after loitering
        - Person near valuable objects
        """
        indicators = []
        theft_score = 0.0
        
        # Loitering near valuable items
        if loitering_time > 60 and near_valuable_objects:
            theft_score += 0.4
            indicators.append('loitering_near_valuables')
        
        # Rapid exit after loitering
        if loitering_time > 30 and speed > 3.0:
            theft_score += 0.3
            indicators.append('rapid_exit_after_loitering')
        
        # Person near valuable objects
        if near_valuable_objects and speed < 0.5:  # Standing still near object
            theft_score += 0.3
            indicators.append('stationary_near_object')
        
        if theft_score >= 0.5:
            return {
                'type': 'potential_theft',
                'track_id': int(track_id),
                'theft_score': float(theft_score),
                'indicators': indicators,
                'severity': 'high',
                'description': f'Potential theft detected - {", ".join(indicators)}'
            }
        
        return None


class SecurityZone:
    """Represents a security zone for monitoring."""
    
    def __init__(self, name: str, polygon: List[Tuple[int, int]], zone_type: str = "restricted"):
        """
        Initialize a security zone.
        
        Args:
            name: Zone identifier
            polygon: List of (x, y) points defining the zone polygon
            zone_type: Type of zone ('restricted', 'entry', 'exit', 'monitoring', 'counting')
        """
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.zone_type = zone_type
        self.objects_inside = set()
        self.entry_count = 0
        self.exit_count = 0
        self.loitering_objects = {}  # track_id -> frames_inside
        self.object_count_history = []  # For counting zones
        
    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the zone."""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0
    
    def update_object(self, track_id: int, center: Tuple[float, float], loitering_threshold: int = 90):
        """Update zone state for an object."""
        is_inside = self.is_point_inside(center)
        was_inside = track_id in self.objects_inside
        
        if is_inside and not was_inside:
            # Object entered zone
            self.objects_inside.add(track_id)
            self.entry_count += 1
            return "entry"
        elif is_inside and was_inside:
            # Object still inside - check for loitering
            self.loitering_objects[track_id] = self.loitering_objects.get(track_id, 0) + 1
            if self.loitering_objects[track_id] >= loitering_threshold:
                return "loitering"
            return "inside"
        elif not is_inside and was_inside:
            # Object exited zone
            self.objects_inside.discard(track_id)
            self.exit_count += 1
            if track_id in self.loitering_objects:
                del self.loitering_objects[track_id]
            return "exit"
        return None
    
    def get_current_count(self) -> int:
        """Get current number of objects in zone."""
        return len(self.objects_inside)


class SecurityTracker:
    """Enhanced security-focused object tracker with advanced features."""
    
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        tracker_config: str = "botsort.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        agnostic_nms: bool = False,
        reid_enabled: bool = False,
        half_precision: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the security tracker with advanced Ultralytics features.
        
        Args:
            model_name: YOLO model to use (yolo11n.pt, yolo11n-seg.pt, yolo11n-pose.pt)
            tracker_config: Tracker configuration ('botsort.yaml' or 'bytetrack.yaml')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to track (None = all classes)
                     Common: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
            max_det: Maximum number of detections per image
            agnostic_nms: Use class-agnostic NMS
            reid_enabled: Enable re-identification for better tracking
            half_precision: Use FP16 inference (faster, less accurate)
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.reid_enabled = reid_enabled
        self.half_precision = half_precision
        self.device = device
        
        # Tracking data
        self.track_history = defaultdict(lambda: deque(maxlen=200))  # Increased history
        self.object_stats = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'total_frames': 0,
            'zones_visited': set(),
            'lines_crossed': set(),
            'class_name': None,
            'speed_history': deque(maxlen=30),
            'direction': None,
            'total_distance': 0.0
        })
        self.alerts = []
        
        # Security zones and lines
        self.zones: List[SecurityZone] = []
        self.lines: List[LineCrossing] = []
        
        # Behavior analyzer for suspicious activity detection
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # Performance metrics
        self.fps_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Frame rate for speed calculation
        self.fps = 30.0  # Will be updated from video
        
        # Valuable objects to monitor (for theft detection)
        self.valuable_object_classes = ['handbag', 'backpack', 'suitcase', 'laptop', 'cell phone', 'purse', 'knife']
    
    def add_zone(self, zone: SecurityZone):
        """Add a security zone to monitor."""
        self.zones.append(zone)
    
    def add_line_crossing(self, line: LineCrossing):
        """Add a line crossing detector."""
        self.lines.append(line)
    
    def add_rectangular_zone(
        self,
        name: str,
        x1: int, y1: int, x2: int, y2: int,
        zone_type: str = "restricted"
    ):
        """Convenience method to add a rectangular zone."""
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        zone = SecurityZone(name, polygon, zone_type)
        self.add_zone(zone)
        return zone
    
    def calculate_speed(self, track_id: int, pixels_per_meter: float = 1.0) -> float:
        """Calculate speed of an object in m/s."""
        track = self.track_history[track_id]
        if len(track) < 2:
            return 0.0
        
        # Get last two positions
        pos1 = np.array(track[-2])
        pos2 = np.array(track[-1])
        
        # Calculate distance in pixels
        distance_pixels = np.linalg.norm(pos2 - pos1)
        
        # Convert to meters
        distance_meters = distance_pixels / pixels_per_meter
        
        # Calculate speed (distance per frame * fps = m/s)
        speed = distance_meters * self.fps
        
        return speed
    
    def calculate_direction(self, track_id: int) -> Optional[str]:
        """Calculate movement direction (N, S, E, W, NE, NW, SE, SW)."""
        track = self.track_history[track_id]
        if len(track) < 5:
            return None
        
        # Use last 5 points to determine direction
        recent_points = list(track)[-5:]
        start = np.array(recent_points[0])
        end = np.array(recent_points[-1])
        
        # Calculate angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if abs(dx) < 5 and abs(dy) < 5:
            return None  # Not moving
        
        angle = np.arctan2(-dy, dx) * 180 / np.pi  # Negative y because image coordinates
        
        # Convert to direction
        if -22.5 <= angle < 22.5:
            return "E"
        elif 22.5 <= angle < 67.5:
            return "SE"
        elif 67.5 <= angle < 112.5:
            return "S"
        elif 112.5 <= angle < 157.5:
            return "SW"
        elif angle >= 157.5 or angle < -157.5:
            return "W"
        elif -157.5 <= angle < -112.5:
            return "NW"
        elif -112.5 <= angle < -67.5:
            return "N"
        elif -67.5 <= angle < -22.5:
            return "NE"
        
        return None
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process a single frame with all Ultralytics tracking features."""
        start_time = time.time()
        
        # Perform tracking with all available parameters
        results = self.model.track(
            frame,
            persist=True,  # Maintain tracking across frames
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            half=self.half_precision,
            device=self.device,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Extract tracking data
        tracking_data = {
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'zone_events': [],
            'line_crossings': [],
            'alerts': []
        }
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
            
            # Process each detection
            for box, track_id, conf, cls_id, cls_name in zip(
                boxes, track_ids, confidences, class_ids, class_names
            ):
                x, y, w, h = box
                center_x = float(x)
                center_y = float(y)
                
                # Get previous position for speed/direction calculation
                prev_pos = None
                if len(self.track_history[track_id]) > 0:
                    prev_pos = self.track_history[track_id][-1]
                
                # Update track history
                self.track_history[track_id].append((center_x, center_y))
                
                # Update object statistics
                stats = self.object_stats[track_id]
                if stats['first_seen'] is None:
                    stats['first_seen'] = frame_number
                    stats['class_name'] = cls_name
                stats['last_seen'] = frame_number
                stats['total_frames'] += 1
                
                # Calculate speed and direction
                speed = self.calculate_speed(track_id)
                stats['speed_history'].append(speed)
                direction = self.calculate_direction(track_id)
                if direction:
                    stats['direction'] = direction
                
                # Calculate distance traveled
                if prev_pos:
                    distance = np.linalg.norm(np.array([center_x, center_y]) - np.array(prev_pos))
                    stats['total_distance'] += distance
                
                # Check line crossings
                for line in self.lines:
                    crossing = line.check_crossing(track_id, (center_x, center_y), prev_pos)
                    if crossing:
                        stats['lines_crossed'].add(line.name)
                        event_data = {
                            'event': 'line_crossing',
                            'line': line.name,
                            'direction': crossing,
                            'track_id': int(track_id),
                            'object_class': cls_name,
                            'frame': frame_number,
                            'timestamp': datetime.now().isoformat()
                        }
                        tracking_data['line_crossings'].append(event_data)
                        
                        # Generate alert for line crossings
                        alert = {
                            'type': 'line_crossing',
                            'line': line.name,
                            'direction': crossing,
                            'track_id': int(track_id),
                            'object_class': cls_name,
                            'frame': frame_number,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'medium'
                        }
                        self.alerts.append(alert)
                        tracking_data['alerts'].append(alert)
                
                # Check zones
                for zone in self.zones:
                    event = zone.update_object(track_id, (center_x, center_y))
                    if event:
                        stats['zones_visited'].add(zone.name)
                        
                        # Generate alert for restricted zone entries
                        if event == "entry" and zone.zone_type == "restricted":
                            alert = {
                                'type': 'zone_entry',
                                'zone': zone.name,
                                'track_id': int(track_id),
                                'object_class': cls_name,
                                'frame': frame_number,
                                'timestamp': datetime.now().isoformat(),
                                'severity': 'high'
                            }
                            self.alerts.append(alert)
                            tracking_data['alerts'].append(alert)
                        
                        # Generate alert for loitering
                        elif event == "loitering":
                            alert = {
                                'type': 'loitering',
                                'zone': zone.name,
                                'track_id': int(track_id),
                                'object_class': cls_name,
                                'frame': frame_number,
                                'duration_frames': zone.loitering_objects[track_id],
                                'timestamp': datetime.now().isoformat(),
                                'severity': 'medium'
                            }
                            self.alerts.append(alert)
                            tracking_data['alerts'].append(alert)
                        
                        tracking_data['zone_events'].append({
                            'event': event,
                            'zone': zone.name,
                            'track_id': int(track_id),
                            'object_class': cls_name
                        })
                
                # Calculate average speed
                avg_speed = np.mean(stats['speed_history']) if len(stats['speed_history']) > 0 else 0.0
                
                # BEHAVIOR ANALYSIS: Check for suspicious activities
                # Check proximity between objects
                for other_track_id, other_stats in self.object_stats.items():
                    if other_track_id != track_id and other_stats['class_name']:
                        other_pos = None
                        if len(self.track_history[other_track_id]) > 0:
                            other_pos = self.track_history[other_track_id][-1]
                        
                        if other_pos:
                            proximity_alert = self.behavior_analyzer.check_proximity(
                                track_id, (center_x, center_y),
                                other_track_id, other_pos,
                                cls_name, other_stats['class_name']
                            )
                            
                            if proximity_alert:
                                alert_data = {
                                    **proximity_alert,
                                    'frame': frame_number,
                                    'timestamp': datetime.now().isoformat()
                                }
                                self.alerts.append(alert_data)
                                tracking_data['alerts'].append(alert_data)
                
                # Check for rapid movement
                rapid_movement = self.behavior_analyzer.detect_rapid_movement(avg_speed, track_id)
                if rapid_movement:
                    alert_data = {
                        **rapid_movement,
                        'frame': frame_number,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert_data)
                    tracking_data['alerts'].append(alert_data)
                
                # Check for sudden stops
                sudden_stop = self.behavior_analyzer.detect_sudden_stop(stats['speed_history'])
                if sudden_stop:
                    alert_data = {
                        **sudden_stop,
                        'track_id': int(track_id),
                        'frame': frame_number,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert_data)
                    tracking_data['alerts'].append(alert_data)
                
                # Check for erratic movement
                erratic = self.behavior_analyzer.detect_erratic_movement(self.track_history[track_id])
                if erratic:
                    alert_data = {
                        **erratic,
                        'track_id': int(track_id),
                        'frame': frame_number,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert_data)
                    tracking_data['alerts'].append(alert_data)
                
                # Check for theft indicators
                loitering_time = 0
                for zone in self.zones:
                    if track_id in zone.loitering_objects:
                        loitering_time = max(loitering_time, zone.loitering_objects[track_id])
                
                # Check if person is near valuable objects
                near_valuables = False
                if cls_name == 'person':
                    for other_track_id, other_stats in self.object_stats.items():
                        if other_stats['class_name'] in self.valuable_object_classes:
                            other_pos = None
                            if len(self.track_history[other_track_id]) > 0:
                                other_pos = self.track_history[other_track_id][-1]
                            if other_pos:
                                distance = np.linalg.norm(np.array([center_x, center_y]) - np.array(other_pos))
                                if distance < self.behavior_analyzer.proximity_threshold:
                                    near_valuables = True
                                    break
                
                theft_alert = self.behavior_analyzer.detect_theft_indicators(
                    track_id, cls_name, avg_speed,
                    stats['zones_visited'], loitering_time, near_valuables
                )
                
                if theft_alert:
                    alert_data = {
                        **theft_alert,
                        'frame': frame_number,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert_data)
                    tracking_data['alerts'].append(alert_data)
                
                # Check for violence using pose estimation (if pose model is available)
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    try:
                        keypoints = results[0].keypoints.xy.cpu().numpy()
                        if len(keypoints) > 0:
                            # Find keypoints for this track_id
                            for i, tid in enumerate(track_ids):
                                if tid == track_id and i < len(keypoints):
                                    pose_keypoints = keypoints[i]
                                    violence_analysis = self.behavior_analyzer.analyze_pose_for_violence(pose_keypoints)
                                    
                                    if violence_analysis['aggressive_pose']:
                                        violence_alert = {
                                            'type': 'violence_detected',
                                            'track_id': int(track_id),
                                            'violence_score': violence_analysis['violence_score'],
                                            'indicators': violence_analysis['indicators'],
                                            'severity': 'high',
                                            'description': f'Aggressive pose detected - possible violence: {", ".join(violence_analysis["indicators"])}',
                                            'frame': frame_number,
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        self.alerts.append(violence_alert)
                                        tracking_data['alerts'].append(violence_alert)
                                    break
                    except Exception:
                        pass  # Continue if pose analysis fails
                
                tracking_data['detections'].append({
                    'track_id': int(track_id),
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'center': [center_x, center_y],
                    'confidence': float(conf),
                    'class': cls_name,
                    'class_id': int(cls_id),
                    'speed': float(avg_speed),
                    'direction': stats['direction']
                })
        
        # Calculate FPS
        if len(self.processing_times) > 0:
            avg_fps = 1.0 / np.mean(self.processing_times)
            self.fps_history.append(avg_fps)
            tracking_data['fps'] = avg_fps
        
        return tracking_data, results[0]
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw security zones on frame."""
        for zone in self.zones:
            # Choose color based on zone type
            colors = {
                'restricted': (0, 0, 255),  # Red
                'entry': (0, 255, 255),     # Yellow
                'exit': (255, 0, 255),      # Magenta
                'monitoring': (0, 255, 0),  # Green
                'counting': (255, 165, 0)   # Orange
            }
            color = colors.get(zone.zone_type, (255, 255, 255))
            
            # Draw polygon
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=color, thickness=2)
            
            # Fill with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone.polygon], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Add zone label with count
            if len(zone.polygon) > 0:
                centroid = zone.polygon.mean(axis=0).astype(int)
                count = zone.get_current_count()
                label = f"{zone.name} ({zone.zone_type}) - Count: {count}"
                cv2.putText(
                    frame, label,
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
        
        return frame
    
    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw crossing lines on frame."""
        for line in self.lines:
            cv2.line(frame, tuple(line.point1), tuple(line.point2), (255, 255, 0), 2)
            
            # Add label
            mid_point = ((line.point1[0] + line.point2[0]) // 2, 
                        (line.point1[1] + line.point2[1]) // 2)
            label = f"{line.name} (F:{line.crossings['forward']} B:{line.crossings['backward']})"
            cv2.putText(
                frame, label,
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 2
            )
        
        return frame
    
    def draw_trajectories(self, frame: np.ndarray, max_length: int = 50) -> np.ndarray:
        """Draw object trajectories with speed and direction indicators."""
        for track_id, track in self.track_history.items():
            if len(track) < 2:
                continue
            
            # Get recent points
            recent_track = list(track)[-max_length:]
            points = np.array(recent_track, dtype=np.int32)
            
            # Draw trajectory with gradient (newer = brighter)
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = (
                    int(230 * alpha),
                    int(230 * alpha),
                    int(255 * alpha)
                )
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, 2)
            
            # Draw current position with speed indicator
            if len(points) > 0:
                current_pos = tuple(points[-1])
                cv2.circle(frame, current_pos, 5, (0, 255, 0), -1)
                
                # Draw speed and direction if available
                stats = self.object_stats[track_id]
                if stats['speed_history'] and len(stats['speed_history']) > 0:
                    avg_speed = np.mean(stats['speed_history'])
                    direction = stats['direction'] or "?"
                    
                    # Draw text above object
                    text = f"ID:{track_id} {stats['class_name']} {avg_speed:.1f}m/s {direction}"
                    text_pos = (current_pos[0] - 50, current_pos[1] - 15)
                    cv2.putText(
                        frame, text,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1
                    )
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray, tracking_data: Dict) -> np.ndarray:
        """Draw comprehensive statistics overlay on frame."""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 30
        line_height = 25
        
        # Frame info
        cv2.putText(
            frame, f"Frame: {tracking_data['frame_number']}",
            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        y_offset += line_height
        
        # FPS
        if 'fps' in tracking_data:
            cv2.putText(
                frame, f"FPS: {tracking_data['fps']:.1f}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += line_height
        
        # Active objects
        active_objects = len(self.track_history)
        cv2.putText(
            frame, f"Active Objects: {active_objects}",
            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        y_offset += line_height
        
        # Zone statistics
        if self.zones:
            total_entries = sum(zone.entry_count for zone in self.zones)
            total_exits = sum(zone.exit_count for zone in self.zones)
            total_current = sum(zone.get_current_count() for zone in self.zones)
            cv2.putText(
                frame, f"Zones: {total_entries} entries, {total_exits} exits, {total_current} current",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            y_offset += line_height
        
        # Line crossing statistics
        if self.lines:
            total_crossings = sum(sum(line.crossings.values()) for line in self.lines)
            cv2.putText(
                frame, f"Line Crossings: {total_crossings}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            y_offset += line_height
        
        # Alerts count
        if self.alerts:
            high_alerts = len([a for a in self.alerts if a.get('severity') == 'high'])
            cv2.putText(
                frame, f"Alerts: {len(self.alerts)} (High: {high_alerts})",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
        
        return frame
    
    def export_tracking_data(self, output_path: str, format: str = "json"):
        """Export tracking statistics and alerts to JSON or CSV."""
        if format.lower() == "json":
            export_data = {
                'summary': {
                    'total_objects_tracked': len(self.object_stats),
                    'total_alerts': len(self.alerts),
                    'zones_monitored': len(self.zones),
                    'lines_monitored': len(self.lines),
                    'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
                    'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0
                },
                'zone_statistics': [
                    {
                        'name': zone.name,
                        'type': zone.zone_type,
                        'entries': zone.entry_count,
                        'exits': zone.exit_count,
                        'current_objects': zone.get_current_count()
                    }
                    for zone in self.zones
                ],
                'line_statistics': [
                    {
                        'name': line.name,
                        'forward_crossings': line.crossings['forward'],
                        'backward_crossings': line.crossings['backward']
                    }
                    for line in self.lines
                ],
                'object_statistics': {
                    str(track_id): {
                        'class': stats['class_name'],
                        'first_seen_frame': stats['first_seen'],
                        'last_seen_frame': stats['last_seen'],
                        'total_frames': stats['total_frames'],
                        'zones_visited': list(stats['zones_visited']),
                        'lines_crossed': list(stats['lines_crossed']),
                        'average_speed': float(np.mean(stats['speed_history'])) if stats['speed_history'] else 0.0,
                        'direction': stats['direction'],
                        'total_distance': float(stats['total_distance'])
                    }
                    for track_id, stats in self.object_stats.items()
                },
                'alerts': self.alerts[-500:]  # Last 500 alerts
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format.lower() == "csv":
            # Export detections to CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'track_id', 'class', 'first_seen', 'last_seen', 'total_frames',
                    'zones_visited', 'lines_crossed', 'avg_speed', 'direction', 'total_distance'
                ])
                
                for track_id, stats in self.object_stats.items():
                    writer.writerow([
                        track_id,
                        stats['class_name'],
                        stats['first_seen'],
                        stats['last_seen'],
                        stats['total_frames'],
                        ','.join(stats['zones_visited']),
                        ','.join(stats['lines_crossed']),
                        np.mean(stats['speed_history']) if stats['speed_history'] else 0.0,
                        stats['direction'] or '',
                        stats['total_distance']
                    ])
        
        print(f"Tracking data exported to: {output_path}")


def track_objects_security(
    video_path: str,
    model_name: str = "yolo11n.pt",
    tracker_config: str = "botsort.yaml",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
    show_video: bool = True,
    save_output: bool = True,
    zones: Optional[List[Dict]] = None,
    lines: Optional[List[Dict]] = None,
    export_data: bool = True,
    export_format: str = "json",
    use_stream: bool = False,
    email_alerts: Optional[Dict] = None
):
    """
    Track objects in a video with comprehensive security-focused features.
    
    Args:
        video_path: Path to input video or stream URL
        model_name: YOLO model to use
        tracker_config: Tracker config ('botsort.yaml' or 'bytetrack.yaml')
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for NMS
        classes: List of class IDs to track (None = all)
        max_det: Maximum detections per frame
        show_video: Display video window
        save_output: Save output video
        zones: List of zone definitions [{'name': str, 'points': [(x,y),...], 'type': str}]
        lines: List of line definitions [{'name': str, 'point1': (x,y), 'point2': (x,y), 'direction': str}]
        export_data: Export tracking statistics
        export_format: Export format ('json' or 'csv')
        use_stream: Use Ultralytics stream mode for processing
        email_alerts: Dict with email config {'from_email': str, 'password': str, 'to_email': str}
    """
    # Initialize tracker
    tracker = SecurityTracker(
        model_name=model_name,
        tracker_config=tracker_config,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        max_det=max_det,
        reid_enabled=False
    )
    
    # Add zones if provided
    if zones:
        for zone_def in zones:
            zone = SecurityZone(
                name=zone_def['name'],
                polygon=zone_def['points'],
                zone_type=zone_def.get('type', 'monitoring')
            )
            tracker.add_zone(zone)
    
    # Add lines if provided
    if lines:
        for line_def in lines:
            line = LineCrossing(
                name=line_def['name'],
                point1=line_def['point1'],
                point2=line_def['point2'],
                direction=line_def.get('direction', 'both')
            )
            tracker.add_line_crossing(line)
    
    # Initialize SecurityAlarm if email alerts requested
    security_alarm = None
    if email_alerts and SECURITY_ALARM_AVAILABLE:
        try:
            security_alarm = SecurityAlarm(
                show=False,
                model=model_name,
                records=1  # Number of detections to trigger email
            )
            security_alarm.authenticate(
                email_alerts['from_email'],
                email_alerts['password'],
                email_alerts['to_email']
            )
            print("Email alerts enabled")
        except Exception as e:
            print(f"Warning: Could not initialize email alerts: {e}")
            security_alarm = None
    
    # Use stream mode if requested
    if use_stream:
        print("Using Ultralytics stream mode...")
        results = tracker.model.track(
            source=str(video_path),
            save=save_output,
            show=show_video,
            tracker=tracker_config,
            conf=conf_threshold,
            classes=classes,
            persist=True,
            verbose=True
        )
        # Stream mode handles everything, so we return early
        return
    
    # Open video
    video_path_obj = Path(video_path)
    if not video_path_obj.exists() and not video_path.startswith(('http', 'rtsp', 'rtmp')):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    tracker.fps = fps  # Update tracker FPS for speed calculation
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"Tracking classes: {classes if classes else 'All'}")
    print(f"Zones configured: {len(tracker.zones)}")
    print(f"Lines configured: {len(tracker.lines)}")
    
    # Setup video writer
    out = None
    if save_output:
        output_path = video_path_obj.parent / f"{video_path_obj.stem}_security_tracked{video_path_obj.suffix}" if video_path_obj.exists() else Path("output_security_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output: {output_path}")
    
    frame_count = 0
    paused = False
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save screenshot")
    print("\nStarting security tracking...\n")
    
    try:
        while cap.isOpened():
            if not paused:
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                
                # Process frame
                tracking_data, results = tracker.process_frame(frame, frame_count)
                
                # Process with SecurityAlarm if enabled
                if security_alarm:
                    try:
                        security_alarm.process(frame)
                    except:
                        pass  # Continue even if alarm fails
                
                # Get annotated frame
                annotated_frame = results.plot()
                
                # Draw zones
                annotated_frame = tracker.draw_zones(annotated_frame)
                
                # Draw lines
                annotated_frame = tracker.draw_lines(annotated_frame)
                
                # Draw trajectories
                annotated_frame = tracker.draw_trajectories(annotated_frame)
                
                # Draw statistics
                annotated_frame = tracker.draw_statistics(annotated_frame, tracking_data)
                
                # Draw alerts
                if tracking_data['alerts']:
                    alert_y = height - 30
                    for alert in tracking_data['alerts'][-3:]:  # Show last 3 alerts
                        alert_text = f"ALERT: {alert['type'].upper()} - {alert.get('object_class', 'Unknown')}"
                        cv2.putText(
                            annotated_frame, alert_text,
                            (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )
                        alert_y -= 30
                
                # Save frame
                if save_output and out is not None:
                    out.write(annotated_frame)
                
                # Display
                if show_video:
                    cv2.imshow("Security Tracking", annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif key == ord('s'):
                        screenshot_path = video_path_obj.parent / f"screenshot_frame_{frame_count}.jpg" if video_path_obj.exists() else Path(f"screenshot_frame_{frame_count}.jpg")
                        cv2.imwrite(str(screenshot_path), annotated_frame)
                        print(f"Screenshot saved: {screenshot_path}")
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Resumed")
            
            # Progress update
            if frame_count % 30 == 0 and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Objects: {len(tracker.track_history)} | Alerts: {len(tracker.alerts)}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Export data
        if export_data:
            if export_format.lower() == "json":
                export_path = video_path_obj.parent / f"{video_path_obj.stem}_tracking_data.json" if video_path_obj.exists() else Path("tracking_data.json")
            else:
                export_path = video_path_obj.parent / f"{video_path_obj.stem}_tracking_data.csv" if video_path_obj.exists() else Path("tracking_data.csv")
            tracker.export_tracking_data(str(export_path), format=export_format)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRACKING SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {frame_count}")
        print(f"Unique objects tracked: {len(tracker.object_stats)}")
        print(f"Total alerts generated: {len(tracker.alerts)}")
        print(f"Average FPS: {np.mean(tracker.fps_history):.2f}" if tracker.fps_history else "N/A")
        
        if tracker.zones:
            print("\nZone Statistics:")
            for zone in tracker.zones:
                print(f"  {zone.name} ({zone.zone_type}):")
                print(f"    Entries: {zone.entry_count}, Exits: {zone.exit_count}")
                print(f"    Current objects: {zone.get_current_count()}")
        
        if tracker.lines:
            print("\nLine Crossing Statistics:")
            for line in tracker.lines:
                print(f"  {line.name}:")
                print(f"    Forward: {line.crossings['forward']}, Backward: {line.crossings['backward']}")
        
        if save_output:
            print(f"\nOutput video saved: {output_path}")


def track_multiple_streams(
    video_sources: List[str],
    model_name: str = "yolo11n.pt",
    tracker_config: str = "botsort.yaml"
):
    """
    Track objects in multiple video streams simultaneously using threading.
    
    Args:
        video_sources: List of video paths or stream URLs
        model_name: YOLO model to use
        tracker_config: Tracker configuration
    """
    def run_tracker(source: str):
        """Run tracker in a separate thread."""
        print(f"Starting tracker for: {source}")
        try:
            track_objects_security(
                video_path=source,
                model_name=model_name,
                tracker_config=tracker_config,
                show_video=True,
                save_output=True,
                export_data=True
            )
        except Exception as e:
            print(f"Error processing {source}: {e}")
    
    threads = []
    for source in video_sources:
        thread = threading.Thread(target=run_tracker, args=(source,), daemon=True)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # Get video path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    video_path = project_root / "macheton.mp4"
    
    print("=" * 60)
    print("Ultralytics YOLO Advanced Security Tracking System")
    print("=" * 60)
    print()
    
    # Example: Track only people (class 0) and vehicles (classes 2, 3, 5, 7)
    # Set to None to track all classes
    security_classes = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
    
    # Define security zones (adjust coordinates based on your video)
    # Format: {'name': str, 'points': [(x1,y1), (x2,y2), ...], 'type': str}
    # Zone types: 'restricted', 'entry', 'exit', 'monitoring', 'counting'
    security_zones = [
        # Example: Add zones programmatically after getting video dimensions
        # Uncomment and adjust coordinates:
        # {
        #     'name': 'Restricted Area',
        #     'points': [(100, 100), (400, 100), (400, 300), (100, 300)],
        #     'type': 'restricted'
        # },
        # {
        #     'name': 'Entry Zone',
        #     'points': [(0, 0), (200, 0), (200, 200), (0, 200)],
        #     'type': 'entry'
        # }
    ]
    
    # Define crossing lines (adjust coordinates based on your video)
    # Format: {'name': str, 'point1': (x1,y1), 'point2': (x2,y2), 'direction': str}
    # Direction: 'forward', 'backward', or 'both'
    crossing_lines = [
        # Example: Add lines programmatically
        # {
        #     'name': 'Main Entrance',
        #     'point1': (100, 200),
        #     'point2': (500, 200),
        #     'direction': 'both'
        # }
    ]
    
    # Optional: Email alerts configuration
    # email_config = {
    #     'from_email': 'your_email@gmail.com',
    #     'password': 'your_app_password',
    #     'to_email': 'recipient@gmail.com'
    # }
    email_config = None
    
    try:
        # track_objects_security(
        #     video_path=str(video_path),
        #     model_name="yolo11n-pose.pt",  # or 'yolo11n-seg.pt', 'yolo11n-pose.pt'
        #     tracker_config="botsort.yaml",  # or "bytetrack.yaml"
        #     conf_threshold=0.25,
        #     iou_threshold=0.7,
        #     classes=security_classes,  # None for all classes
        #     max_det=300,
        #     show_video=True,
        #     save_output=True,
        #     zones=security_zones if security_zones else None,
        #     lines=crossing_lines if crossing_lines else None,
        #     export_data=True,
        #     export_format="json",  # or "csv"
        #     use_stream=False,  # Set True to use Ultralytics stream mode
        #     email_alerts=email_config
        # )
        track_objects_security(
            video_path=str(video_path),
            model_name="yolo11n-pose.pt",  # ← IMPORTANT: Use pose model!
            tracker_config="botsort.yaml",
            conf_threshold=0.25,
            classes=[0],  # Track people only
            show_video=True,
            save_output=True,
            export_data=True,
            export_format="json"
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Looking for video at: {video_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
