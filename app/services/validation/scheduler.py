"""
Validation Worker Scheduler - Production Grade
Version: 2.0 (FastAPI-Integrated, Background Tasks)

Responsibilities:
- Run validation workers automatically
- Maintain correct execution order
- Handle errors gracefully
- Provide health monitoring
- Never block main application
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from app.utils.logger import log_info, log_warning, log_error, log_debug


# ============================================================
# SCHEDULER CONFIGURATION
# ============================================================

@dataclass
class SchedulerConfig:
    """
    Immutable scheduler configuration.
    """
    
    # Worker execution intervals (seconds)
    validation_interval: int = 60       # Run agentic validation every 60s
    conflict_interval: int = 120        # Run conflict detection every 2 min
    temporal_interval: int = 300        # Run temporal revalidation every 5 min
    
    # Batch sizes (how many chunks per run)
    validation_batch_size: int = 50
    conflict_batch_size: int = 30
    temporal_batch_size: int = 50
    
    # Execution order delay (seconds)
    # Conflict detection waits for validation to finish
    conflict_delay_after_validation: int = 10
    
    # Health check
    health_check_interval: int = 60     # Check health every 60s
    max_consecutive_failures: int = 5   # Restart after 5 failures
    
    # Graceful shutdown
    shutdown_timeout: int = 30          # Max wait for graceful shutdown
    
    # Enable/disable workers
    enable_validation: bool = True
    enable_conflict: bool = True
    enable_temporal: bool = True


class WorkerStatus(str, Enum):
    """Worker execution status"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class WorkerStats:
    """Statistics for a single worker"""
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    last_run_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    total_runs: int = 0
    total_successes: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_duration_ms: Optional[float] = None
    last_error: Optional[str] = None
    items_processed: int = 0


# ============================================================
# VALIDATION SCHEDULER
# ============================================================

class ValidationScheduler:
    """
    Orchestrates all validation workers as background tasks.
    
    Architecture:
    - Each worker runs in its own asyncio task
    - Workers execute at configurable intervals
    - Execution order: validation → conflict → temporal
    - Errors are isolated (one worker failure doesn't affect others)
    - Health monitoring tracks worker status
    - Graceful shutdown ensures clean termination
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Worker statistics
        self._stats = {
            "agentic_validation": WorkerStats("agentic_validation"),
            "conflict_detection": WorkerStats("conflict_detection"),
            "temporal_revalidation": WorkerStats("temporal_revalidation"),
        }
        
        # System stats
        self._scheduler_started_at: Optional[datetime] = None
        self._last_health_check_at: Optional[datetime] = None
    
    # --------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------
    
    async def start(self) -> None:
        """
        Start the validation scheduler.
        
        This creates background tasks for each enabled worker.
        Safe to call multiple times (idempotent).
        """
        if self._running:
            log_warning("[Scheduler] Already running, ignoring start request")
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._scheduler_started_at = datetime.now(timezone.utc)
        
        log_info(
            f"[Scheduler] Starting validation scheduler | "
            f"validation_interval={self.config.validation_interval}s | "
            f"conflict_interval={self.config.conflict_interval}s | "
            f"temporal_interval={self.config.temporal_interval}s"
        )
        
        # Create background tasks
        if self.config.enable_validation:
            task = asyncio.create_task(self._run_validation_loop())
            task.set_name("agentic_validation_worker")
            self._tasks.append(task)
        else:
            self._stats["agentic_validation"].status = WorkerStatus.DISABLED
        
        if self.config.enable_conflict:
            task = asyncio.create_task(self._run_conflict_loop())
            task.set_name("conflict_detection_worker")
            self._tasks.append(task)
        else:
            self._stats["conflict_detection"].status = WorkerStatus.DISABLED
        
        if self.config.enable_temporal:
            task = asyncio.create_task(self._run_temporal_loop())
            task.set_name("temporal_revalidation_worker")
            self._tasks.append(task)
        else:
            self._stats["temporal_revalidation"].status = WorkerStatus.DISABLED
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        health_task.set_name("health_monitor")
        self._tasks.append(health_task)
        
        log_info(
            f"[Scheduler] Started {len(self._tasks)} background tasks "
            f"(validation: {self.config.enable_validation}, "
            f"conflict: {self.config.enable_conflict}, "
            f"temporal: {self.config.enable_temporal})"
        )
    
    async def stop(self) -> None:
        """
        Stop the validation scheduler gracefully.
        
        Waits for running workers to complete (up to shutdown_timeout).
        Cancels tasks if they don't finish in time.
        """
        if not self._running:
            log_warning("[Scheduler] Not running, ignoring stop request")
            return
        
        log_info("[Scheduler] Initiating graceful shutdown...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Wait for tasks to finish gracefully
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=self.config.shutdown_timeout
                )
                log_info("[Scheduler] All tasks completed gracefully")
            except asyncio.TimeoutError:
                log_warning(
                    f"[Scheduler] Shutdown timeout ({self.config.shutdown_timeout}s), "
                    f"cancelling remaining tasks"
                )
                # Force cancel remaining tasks
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for cancellations
                await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        log_info("[Scheduler] Scheduler stopped")
    
    # --------------------------------------------------------
    # Worker Loops
    # --------------------------------------------------------
    
    async def _run_validation_loop(self) -> None:
        """
        Background loop for agentic validation worker.
        """
        worker_name = "agentic_validation"
        stats = self._stats[worker_name]
        
        log_info(f"[{worker_name}] Worker started")
        
        while self._running:
            try:
                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.validation_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal interval elapsed
                
                # Execute worker
                await self._execute_worker(
                    worker_name=worker_name,
                    worker_func=self._run_agentic_validation,
                    batch_size=self.config.validation_batch_size
                )
                
            except asyncio.CancelledError:
                log_info(f"[{worker_name}] Worker cancelled")
                break
            except Exception as e:
                log_error(f"[{worker_name}] Unexpected error in worker loop: {e}")
                await asyncio.sleep(10)  # Back off on errors
        
        log_info(f"[{worker_name}] Worker stopped")
    
    async def _run_conflict_loop(self) -> None:
        """
        Background loop for conflict detection worker.
        
        Runs after validation worker to ensure validated data exists.
        """
        worker_name = "conflict_detection"
        stats = self._stats[worker_name]
        
        log_info(f"[{worker_name}] Worker started")
        
        # Initial delay to let validation run first
        await asyncio.sleep(self.config.conflict_delay_after_validation)
        
        while self._running:
            try:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.conflict_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
                await self._execute_worker(
                    worker_name=worker_name,
                    worker_func=self._run_conflict_detection,
                    batch_size=self.config.conflict_batch_size
                )
                
            except asyncio.CancelledError:
                log_info(f"[{worker_name}] Worker cancelled")
                break
            except Exception as e:
                log_error(f"[{worker_name}] Unexpected error in worker loop: {e}")
                await asyncio.sleep(10)
        
        log_info(f"[{worker_name}] Worker stopped")
    
    async def _run_temporal_loop(self) -> None:
        """
        Background loop for temporal revalidation worker.
        """
        worker_name = "temporal_revalidation"
        stats = self._stats[worker_name]
        
        log_info(f"[{worker_name}] Worker started")
        
        while self._running:
            try:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.temporal_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
                await self._execute_worker(
                    worker_name=worker_name,
                    worker_func=self._run_temporal_revalidation,
                    batch_size=self.config.temporal_batch_size
                )
                
            except asyncio.CancelledError:
                log_info(f"[{worker_name}] Worker cancelled")
                break
            except Exception as e:
                log_error(f"[{worker_name}] Unexpected error in worker loop: {e}")
                await asyncio.sleep(10)
        
        log_info(f"[{worker_name}] Worker stopped")
    
    # --------------------------------------------------------
    # Worker Execution
    # --------------------------------------------------------
    
    async def _execute_worker(
        self,
        worker_name: str,
        worker_func,
        batch_size: int
    ) -> None:
        """
        Execute a single worker run with error handling and stats tracking.
        """
        stats = self._stats[worker_name]
        stats.status = WorkerStatus.RUNNING
        stats.total_runs += 1
        
        start_time = datetime.now(timezone.utc)
        
        try:
            log_debug(f"[{worker_name}] Starting run #{stats.total_runs}")
            
            # Execute worker function
            result = await worker_func(batch_size)
            
            # Update stats
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stats.status = WorkerStatus.SUCCESS
            stats.last_run_at = start_time
            stats.last_success_at = start_time
            stats.total_successes += 1
            stats.consecutive_failures = 0
            stats.last_duration_ms = duration_ms
            stats.last_error = None
            stats.items_processed = result.get("processed", 0)
            
            log_info(
                f"[{worker_name}] ✅ Run complete | "
                f"processed={result.get('processed', 0)} | "
                f"duration={duration_ms:.0f}ms"
            )
            
        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stats.status = WorkerStatus.FAILED
            stats.last_run_at = start_time
            stats.last_failure_at = start_time
            stats.total_failures += 1
            stats.consecutive_failures += 1
            stats.last_duration_ms = duration_ms
            stats.last_error = str(e)
            
            log_error(
                f"[{worker_name}] ❌ Run failed | "
                f"error={e} | "
                f"consecutive_failures={stats.consecutive_failures}"
            )
            
            # Check if we should disable this worker
            if stats.consecutive_failures >= self.config.max_consecutive_failures:
                log_error(
                    f"[{worker_name}] Reached max consecutive failures "
                    f"({self.config.max_consecutive_failures}), worker may need attention"
                )
    
    # --------------------------------------------------------
    # Worker Function Wrappers
    # --------------------------------------------------------
    
    async def _run_agentic_validation(self, batch_size: int) -> Dict[str, Any]:
        """Wrapper for agentic validation worker"""
        from app.services.validation.agentic_validation_worker import (
            run_agentic_validation
        )
        return await run_agentic_validation(batch_size)
    
    async def _run_conflict_detection(self, batch_size: int) -> Dict[str, Any]:
        """Wrapper for conflict detection worker"""
        from app.services.validation.semantic_conflict_engine import (
            run_semantic_conflict_detection
        )
        return await run_semantic_conflict_detection(batch_size)
    
    async def _run_temporal_revalidation(self, batch_size: int) -> Dict[str, Any]:
        """Wrapper for temporal revalidation worker"""
        from app.services.validation.temporal_revalidation_engine import (
            run_temporal_revalidation
        )
        return await run_temporal_revalidation(batch_size)
    
    # --------------------------------------------------------
    # Health Monitoring
    # --------------------------------------------------------
    
    async def _health_monitor_loop(self) -> None:
        """
        Background health monitor.
        Checks worker status and logs warnings for unhealthy workers.
        """
        log_info("[HealthMonitor] Health monitor started")
        
        while self._running:
            try:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.health_check_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
                self._last_health_check_at = datetime.now(timezone.utc)
                
                # Check each worker
                unhealthy_workers = []
                for worker_name, stats in self._stats.items():
                    if stats.status == WorkerStatus.DISABLED:
                        continue
                    
                    # Check for stalled workers
                    if stats.last_run_at:
                        time_since_last_run = (
                            datetime.now(timezone.utc) - stats.last_run_at
                        ).total_seconds()
                        
                        # Get expected interval for this worker
                        expected_interval = {
                            "agentic_validation": self.config.validation_interval,
                            "conflict_detection": self.config.conflict_interval,
                            "temporal_revalidation": self.config.temporal_interval,
                        }.get(worker_name, 300)
                        
                        if time_since_last_run > expected_interval * 3:
                            unhealthy_workers.append(
                                f"{worker_name} (stalled, {time_since_last_run:.0f}s since last run)"
                            )
                    
                    # Check for consecutive failures
                    if stats.consecutive_failures >= 3:
                        unhealthy_workers.append(
                            f"{worker_name} ({stats.consecutive_failures} consecutive failures)"
                        )
                
                if unhealthy_workers:
                    log_warning(
                        f"[HealthMonitor] Unhealthy workers detected: "
                        f"{', '.join(unhealthy_workers)}"
                    )
                else:
                    log_debug("[HealthMonitor] All workers healthy")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"[HealthMonitor] Error in health check: {e}")
        
        log_info("[HealthMonitor] Health monitor stopped")
    
    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Dict with scheduler and worker stats
        """
        uptime_seconds = 0
        if self._scheduler_started_at:
            uptime_seconds = (
                datetime.now(timezone.utc) - self._scheduler_started_at
            ).total_seconds()
        
        return {
            "scheduler": {
                "running": self._running,
                "started_at": self._scheduler_started_at.isoformat() if self._scheduler_started_at else None,
                "uptime_seconds": round(uptime_seconds, 2),
                "last_health_check_at": self._last_health_check_at.isoformat() if self._last_health_check_at else None,
            },
            "workers": {
                name: {
                    "status": stats.status.value,
                    "last_run_at": stats.last_run_at.isoformat() if stats.last_run_at else None,
                    "last_success_at": stats.last_success_at.isoformat() if stats.last_success_at else None,
                    "total_runs": stats.total_runs,
                    "total_successes": stats.total_successes,
                    "total_failures": stats.total_failures,
                    "consecutive_failures": stats.consecutive_failures,
                    "last_duration_ms": stats.last_duration_ms,
                    "last_error": stats.last_error,
                    "items_processed": stats.items_processed,
                }
                for name, stats in self._stats.items()
            },
            "config": {
                "validation_interval": self.config.validation_interval,
                "conflict_interval": self.config.conflict_interval,
                "temporal_interval": self.config.temporal_interval,
                "validation_batch_size": self.config.validation_batch_size,
                "conflict_batch_size": self.config.conflict_batch_size,
                "temporal_batch_size": self.config.temporal_batch_size,
            },
        }


# ============================================================
# GLOBAL SCHEDULER INSTANCE
# ============================================================

_global_scheduler: Optional[ValidationScheduler] = None


async def start_validation_scheduler(
    config: Optional[SchedulerConfig] = None
) -> ValidationScheduler:
    """
    Start the global validation scheduler.
    
    Args:
        config: Optional custom configuration
    
    Returns:
        The global scheduler instance
    """
    global _global_scheduler
    
    if _global_scheduler is None:
        _global_scheduler = ValidationScheduler(config)
    
    await _global_scheduler.start()
    return _global_scheduler


async def stop_validation_scheduler() -> None:
    """Stop the global validation scheduler."""
    global _global_scheduler
    
    if _global_scheduler:
        await _global_scheduler.stop()


def get_validation_scheduler() -> Optional[ValidationScheduler]:
    """Get the global scheduler instance (if running)."""
    return _global_scheduler
