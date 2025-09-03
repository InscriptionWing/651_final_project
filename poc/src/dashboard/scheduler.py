"""
ETL Scheduler
Scheduled ETL job for regular data aggregation
"""

import schedule
import time
import logging
from datetime import datetime
from threading import Thread
from typing import Optional
import sys
from pathlib import Path

from data_aggregator import DataAggregator
from config import get_dashboard_config

logger = logging.getLogger(__name__)

class ETLScheduler:
    """Scheduler for ETL jobs"""
    
    def __init__(self, aggregator: Optional[DataAggregator] = None):
        self.aggregator = aggregator or DataAggregator()
        self.config = get_dashboard_config()
        self.running = False
        self.thread = None
    
    def run_etl_job(self):
        """Run a single ETL job"""
        try:
            logger.info("Starting scheduled ETL job")
            start_time = datetime.now()
            
            # Run data aggregation
            result = self.aggregator.aggregate_all_data()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.get("status") == "success":
                logger.info(f"ETL job completed successfully in {duration:.2f} seconds")
                logger.info(f"Processed {result.get('output_metrics', {}).get('total_records', 0)} records")
            else:
                logger.error(f"ETL job failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"ETL job exception: {e}")
            logger.exception("ETL job failed with exception")
    
    def start_scheduler(self, interval_minutes: int = 5):
        """Start the ETL scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info(f"Starting ETL scheduler with {interval_minutes} minute intervals")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(self.run_etl_job)
        
        # Also run immediately on startup
        self.run_etl_job()
        
        self.running = True
        
        # Start scheduler in background thread
        self.thread = Thread(target=self._run_scheduler_loop, daemon=True)
        self.thread.start()
        
        logger.info("ETL scheduler started successfully")
    
    def stop_scheduler(self):
        """Stop the ETL scheduler"""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping ETL scheduler")
        self.running = False
        schedule.clear()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("ETL scheduler stopped")
    
    def _run_scheduler_loop(self):
        """Main scheduler loop (runs in background thread)"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def run_once(self):
        """Run ETL job once (for testing or manual execution)"""
        logger.info("Running ETL job once")
        self.run_etl_job()
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            "running": self.running,
            "scheduled_jobs": len(schedule.jobs),
            "next_run": str(schedule.next_run()) if schedule.jobs else None,
            "thread_alive": self.thread.is_alive() if self.thread else False
        }

def main():
    """Main function for running scheduler as standalone script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NDIS Carer Data ETL Scheduler")
    parser.add_argument("--interval", type=int, default=5, 
                       help="ETL job interval in minutes (default: 5)")
    parser.add_argument("--run-once", action="store_true",
                       help="Run ETL job once and exit")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon process")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/etl_scheduler.log', encoding='utf-8')
        ]
    )
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("Starting NDIS Carer Data ETL Scheduler")
    
    try:
        scheduler = ETLScheduler()
        
        if args.run_once:
            # Run once and exit
            scheduler.run_once()
            logger.info("ETL job completed, exiting")
            return
        
        # Start scheduler
        scheduler.start_scheduler(interval_minutes=args.interval)
        
        if args.daemon:
            # Run as daemon
            logger.info("Running as daemon process")
            try:
                while scheduler.running:
                    time.sleep(10)
                    # Print status periodically
                    status = scheduler.get_status()
                    logger.debug(f"Scheduler status: {status}")
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
        else:
            # Interactive mode
            logger.info("ETL Scheduler started. Press Ctrl+C to stop.")
            logger.info("Commands: 'status', 'run', 'stop', 'quit'")
            
            try:
                while True:
                    try:
                        command = input("> ").strip().lower()
                        
                        if command == "status":
                            status = scheduler.get_status()
                            print(f"Scheduler Status: {status}")
                        
                        elif command == "run":
                            print("Running ETL job manually...")
                            scheduler.run_once()
                            print("ETL job completed")
                        
                        elif command == "stop":
                            scheduler.stop_scheduler()
                            print("Scheduler stopped")
                        
                        elif command in ["quit", "exit", "q"]:
                            break
                        
                        elif command == "help":
                            print("Available commands: status, run, stop, quit, help")
                        
                        elif command:
                            print(f"Unknown command: {command}. Type 'help' for available commands.")
                            
                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        break
                        
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
        
        # Cleanup
        scheduler.stop_scheduler()
        logger.info("ETL Scheduler shutdown complete")
        
    except Exception as e:
        logger.error(f"ETL Scheduler failed: {e}")
        logger.exception("ETL Scheduler exception")
        sys.exit(1)

if __name__ == "__main__":
    main()



