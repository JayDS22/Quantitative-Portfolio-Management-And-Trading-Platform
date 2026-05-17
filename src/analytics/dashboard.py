"""Tableau Dashboard Integration."""
import logging

class DashboardManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
    
    async def start(self):
        self.is_running = True
        self.logger.info("Dashboard manager started")
    
    async def stop(self):
        self.is_running = False
        self.logger.info("Dashboard manager stopped")
