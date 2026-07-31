"""
Production WSGI Server Launcher (prod_server.py)
Runs the NeuroCare Patient Portal on a multi-threaded Waitress WSGI production server,
accessible across local network devices and ready for cloud deployment.
"""

from waitress import serve
from app import app
from utils.logger import logger

if __name__ == '__main__':
    logger.info("==================================================")
    logger.info("  STARTING NEUROCARE PRODUCTION WSGI SERVER      ")
    logger.info("  Server URL: http://127.0.0.1:5000               ")
    logger.info("  Local Network Access: http://0.0.0.0:5000       ")
    logger.info("==================================================")
    serve(app, host='0.0.0.0', port=5000, threads=6)
