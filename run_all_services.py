import subprocess
import sys
import time
import signal
import psutil
import os
from typing import List
import logging
import threading
from queue import Queue
import requests
from requests.exceptions import RequestException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        # Получаем текущее окружение и настраиваем его один раз при инициализации
        self.env = os.environ.copy()
        # Добавляем путь к Python
        self.python_path = sys.executable
        # Настраиваем общие параметры для всех процессов
        self.process_params = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
            'bufsize': 1,
            'universal_newlines': True,
            'env': self.env
        }
        logger.info(f"Using Python interpreter: {self.python_path}")
        
        self.services = [
            {
                'name': 'Search Service',
                'command': [self.python_path, 'service.py'],
                'port': 8001,
                'health_url': 'http://127.0.0.1:8001/health',
                'timeout': 60
            },
            {
                'name': 'LLM Service',
                'command': [self.python_path, 'llm_service_mist_gguf.py'],
                'port': 8002,
                'health_url': 'http://127.0.0.1:8002/health',
                'timeout': 300
            },
            {
                'name': 'Router Service',
                'command': [self.python_path, 'router_service.py'],
                'port': 8003,
                'health_url': 'http://127.0.0.1:8003/health',
                'timeout': 60
            }
        ]
        self.output_queues = {}

    def is_port_in_use(self, port: int) -> bool:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return False
        except OSError:
            return True

    def kill_process_on_port(self, port: int):
        if not self.is_port_in_use(port):
            return
            
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                process = psutil.Process(proc.info['pid'])
                for conn in process.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {process.pid} on port {port}")
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    def wait_for_service(self, service: dict) -> bool:
        timeout = service.get('timeout', 60)
        start_time = time.time()
        logger.info(f"Waiting for {service['name']} to start (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(service['health_url'], timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service['name']} is ready after {time.time() - start_time:.1f} seconds")
                    return True
            except RequestException as e:
                if time.time() - start_time > timeout / 2:
                    logger.warning(f"Still waiting for {service['name']}... ({str(e)})")
                time.sleep(2)
        
        logger.error(f"Timeout waiting for {service['name']} after {timeout} seconds")
        return False

    def read_output(self, process: subprocess.Popen, name: str, queue: Queue):
        while True:
            if process.poll() is not None:
                break
            
            output = process.stdout.readline()
            if output:
                queue.put(f"{name} - OUT: {output.strip()}")
            
            error = process.stderr.readline()
            if error:
                queue.put(f"{name} - ERR: {error.strip()}")

    def process_output(self):
        while True:
            for queue in self.output_queues.values():
                try:
                    while not queue.empty():
                        message = queue.get_nowait()
                        logger.info(message)
                except Exception:
                    pass
            time.sleep(0.1)

    def start_services(self):
        logger.info("Starting all services...")
        
        # Очищаем порты
        for service in self.services:
            service['command'][0] = self.python_path
            if self.is_port_in_use(service['port']):
                logger.info(f"Port {service['port']} is in use, cleaning up...")
                self.kill_process_on_port(service['port'])
                time.sleep(1)
        
        # Запускаем все сервисы параллельно
        for service in self.services:
            try:
                logger.info(f"Starting {service['name']}...")
                logger.info(f"Command: {' '.join(service['command'])}")
                
                # Используем предварительно настроенные параметры процесса
                process = subprocess.Popen(
                    service['command'],
                    **self.process_params
                )
                self.processes.append(process)
                
                queue = Queue()
                self.output_queues[service['name']] = queue
                
                thread = threading.Thread(
                    target=self.read_output,
                    args=(process, service['name'], queue),
                    daemon=True
                )
                thread.start()
                
            except Exception as e:
                logger.error(f"Failed to start {service['name']}: {str(e)}")
                self.shutdown()
                sys.exit(1)
        
        # Запускаем обработку вывода
        output_thread = threading.Thread(
            target=self.process_output,
            daemon=True
        )
        output_thread.start()
        
        # Ждем готовности всех сервисов
        logger.info("Waiting for all services to be ready...")
        for service in self.services:
            if not self.wait_for_service(service):
                logger.error(f"{service['name']} failed to start")
                self.shutdown()
                sys.exit(1)
        
        logger.info("All services are running!")

    def shutdown(self):
        logger.info("Shutting down all services...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()
        
        for service in self.services:
            if self.is_port_in_use(service['port']):
                self.kill_process_on_port(service['port'])
        
        logger.info("All services stopped")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    if manager:
        manager.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    manager = None
    
    try:
        # Регистрируем обработчики сигналов
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        manager = ServiceManager()
        manager.start_services()
        
        # Держим процесс активным
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        if manager:
            manager.shutdown()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if manager:
            manager.shutdown()
        sys.exit(1)