"""Progress tracking for model training pipeline."""
import threading
import time
from typing import Dict, Optional
from datetime import datetime


class ProgressTracker:
    """Thread-safe progress tracker for model training."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.lock = threading.Lock()
        self.stages = {
            'data_preprocessing': {'status': 'pending', 'progress': 0, 'message': 'Preparing data...'},
            'outlier_handling': {'status': 'pending', 'progress': 0, 'message': 'Not selected'},
            'feature_selection': {'status': 'pending', 'progress': 0, 'message': 'Not selected'},
            'hyperparameter_search': {'status': 'pending', 'progress': 0, 'message': 'Not selected'},
            'model_training': {'status': 'pending', 'progress': 0, 'message': 'Waiting...'},
            'cross_validation': {'status': 'pending', 'progress': 0, 'message': 'Not selected'},
            'visualization': {'status': 'pending', 'progress': 0, 'message': 'Waiting...'},
        }
        self.overall_progress = 0
        self.current_stage = None
        self.start_time = None
        self.estimated_time_remaining = None
        
    def start(self):
        """Mark the start of processing."""
        with self.lock:
            self.start_time = time.time()
            self.overall_progress = 0
            
    def update_stage(self, stage: str, status: str, progress: float, message: str = None):
        """Update a specific stage's progress."""
        with self.lock:
            if stage in self.stages:
                self.stages[stage]['status'] = status
                self.stages[stage]['progress'] = max(0, min(100, progress))
                if message:
                    self.stages[stage]['message'] = message
                self.current_stage = stage
                self._update_overall_progress()
                
    def set_stage_enabled(self, stage: str, enabled: bool):
        """Mark a stage as enabled or disabled."""
        with self.lock:
            if stage in self.stages:
                if not enabled:
                    self.stages[stage]['status'] = 'skipped'
                    self.stages[stage]['progress'] = 100
                    self.stages[stage]['message'] = 'Not selected'
                else:
                    self.stages[stage]['status'] = 'pending'
                    self.stages[stage]['progress'] = 0
                    # Update message based on stage type when enabling
                    if stage == 'cross_validation':
                        self.stages[stage]['message'] = 'Waiting...'
                    elif stage == 'outlier_handling':
                        self.stages[stage]['message'] = 'Waiting...'
                    elif stage == 'feature_selection':
                        self.stages[stage]['message'] = 'Waiting...'
                    elif stage == 'hyperparameter_search':
                        self.stages[stage]['message'] = 'Waiting...'
                    
    def _update_overall_progress(self):
        """Calculate overall progress based on enabled stages."""
        enabled_stages = [s for s in self.stages.values() if s['status'] != 'skipped']
        if not enabled_stages:
            self.overall_progress = 100
            return
            
        total_weight = len(enabled_stages)
        total_progress = sum(s['progress'] for s in enabled_stages)
        self.overall_progress = total_progress / total_weight if total_weight > 0 else 0
        
        # Estimate time remaining - improved to account for pending stages
        if self.start_time and self.overall_progress > 0:
            elapsed = time.time() - self.start_time
            if self.overall_progress < 100:
                # Count pending stages that haven't started
                pending_stages = [s for s in enabled_stages if s['status'] == 'pending']
                
                # If there are pending stages, don't estimate time remaining
                # This prevents misleading "1s away" estimates when long operations (like hyperparameter search) are still running
                if pending_stages:
                    # Don't estimate when stages are still pending - it's too unreliable
                    self.estimated_time_remaining = None
                else:
                    # All stages are either completed or running - safe to estimate
                    estimated_total = elapsed / (self.overall_progress / 100)
                    self.estimated_time_remaining = max(0, estimated_total - elapsed)
            else:
                self.estimated_time_remaining = 0
        else:
            self.estimated_time_remaining = None
                
    def get_progress(self) -> Dict:
        """Get current progress state."""
        with self.lock:
            return {
                'session_id': self.session_id,
                'overall_progress': round(self.overall_progress, 1),
                'current_stage': self.current_stage,
                'stages': self.stages.copy(),
                'estimated_time_remaining': self.estimated_time_remaining,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0
            }
            
    def complete(self):
        """Mark all stages as complete."""
        with self.lock:
            for stage in self.stages:
                if self.stages[stage]['status'] != 'skipped':
                    self.stages[stage]['status'] = 'completed'
                    self.stages[stage]['progress'] = 100
            self.overall_progress = 100
            self.estimated_time_remaining = 0


# Global progress trackers (session_id -> ProgressTracker)
_progress_trackers: Dict[str, ProgressTracker] = {}
_tracker_lock = threading.Lock()

# Global results storage (session_id -> result dict)
_results: Dict[str, Dict] = {}
_results_lock = threading.Lock()


def get_tracker(session_id: str) -> ProgressTracker:
    """Get or create a progress tracker for a session."""
    with _tracker_lock:
        if session_id not in _progress_trackers:
            _progress_trackers[session_id] = ProgressTracker(session_id)
        return _progress_trackers[session_id]


def remove_tracker(session_id: str):
    """Remove a progress tracker and its result (cleanup)."""
    with _tracker_lock:
        if session_id in _progress_trackers:
            del _progress_trackers[session_id]
    with _results_lock:
        if session_id in _results:
            del _results[session_id]


def set_result(session_id: str, result: Dict):
    """Store the final result for a session."""
    with _results_lock:
        _results[session_id] = result


def get_result(session_id: str) -> Optional[Dict]:
    """Get the final result for a session."""
    with _results_lock:
        return _results.get(session_id)
