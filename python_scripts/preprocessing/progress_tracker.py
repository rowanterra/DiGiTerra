"""Progress tracking for model training pipeline."""
import threading
import time


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
        self.failed = False
        self.error = None
        
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
                
    def get_progress(self) -> dict:
        """Get current progress state."""
        with self.lock:
            return {
                'session_id': self.session_id,
                'failed': self.failed,
                'error': self.error,
                'overall_progress': round(self.overall_progress, 1),
                'current_stage': self.current_stage,
                'stages': self.stages.copy(),
                'estimated_time_remaining': self.estimated_time_remaining,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0
            }
            
    def fail(self, message: str):
        """Mark the run as failed and drive overall progress to the terminal value.

        The progress stream ends only when overall progress reaches the completion
        threshold. A failed run left below the threshold keeps the stream open until the
        one-hour timeout and the error never reaches the client.
        """
        with self.lock:
            self.failed = True
            self.error = message
            stage = self.current_stage or 'model_training'
            if stage in self.stages:
                self.stages[stage]['status'] = 'failed'
                self.stages[stage]['message'] = f'Error: {message}'
            self.overall_progress = 100
            self.estimated_time_remaining = 0

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
_progress_trackers: dict[str, ProgressTracker] = {}
_tracker_lock = threading.Lock()

# Global results storage (session_id -> result dict)
_results: dict[str, dict] = {}
_results_lock = threading.Lock()


# A run that is never polled would otherwise keep its tracker and result forever.
MAX_TRACKED_SESSIONS = 256


def get_tracker(session_id: str) -> ProgressTracker:
    """Get or create a progress tracker for a session."""
    with _tracker_lock:
        if session_id not in _progress_trackers:
            while len(_progress_trackers) >= MAX_TRACKED_SESSIONS:
                oldest = next(iter(_progress_trackers))
                del _progress_trackers[oldest]
                with _results_lock:
                    _results.pop(oldest, None)
            _progress_trackers[session_id] = ProgressTracker(session_id)
        return _progress_trackers[session_id]


def peek_tracker(session_id: str) -> ProgressTracker | None:
    """Return an existing tracker, or None. Never creates one.

    The progress endpoint takes a session id from the URL. Creating a tracker on
    demand there lets any request open a stream that runs until its timeout, which
    on a single-worker server blocks every other request.
    """
    with _tracker_lock:
        return _progress_trackers.get(session_id)


def remove_tracker(session_id: str):
    """Remove a progress tracker and its result (cleanup)."""
    with _tracker_lock:
        if session_id in _progress_trackers:
            del _progress_trackers[session_id]
    with _results_lock:
        if session_id in _results:
            del _results[session_id]


def set_result(session_id: str, result: dict):
    """Store the final result for a session."""
    with _results_lock:
        _results[session_id] = result


def get_result(session_id: str) -> dict | None:
    """Get the final result for a session."""
    with _results_lock:
        return _results.get(session_id)
