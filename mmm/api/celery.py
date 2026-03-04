import logging
from typing import TYPE_CHECKING

import logfire

try:
    import kombu
    from celery import Celery
    from celery.signals import worker_init
except ImportError:
    if not TYPE_CHECKING:
        Celery, kombu, worker_init = None, None, None

from mmm.api.WorkerState import ws

if worker_init is not None:

    @worker_init.connect
    def init_worker(*args, **kwargs):
        logfire.configure(service_name="m3_worker")
        logfire.instrument_celery()

    redis_url = ws.settings.redis_url
    if redis_url.startswith("rediss://"):
        redis_url += "?ssl_cert_reqs=CERT_OPTIONAL"
    app = Celery("tasks", backend=redis_url, broker=redis_url)
    app.conf.result_backend_transport_options = {"global_keyprefix": ws.settings.redis_backend_prefix}
    app.conf.broker_transport_options = {"global_keyprefix": ws.settings.redis_backend_prefix}
    app.conf.broker_connection_retry_on_startup = False
    app.conf.task_serializer, app.conf.result_serializer = "pickle", "pickle"
    app.conf.task_default_queue = "m3worker"
    app.conf.accept_content = [
        "application/json",
        "application/x-msgpack",
        "application/x-python-serialize",
    ]
else:

    class app:
        @staticmethod
        def task(f):
            return f


@app.task
def get_number_of_celery_workers() -> int:
    """
    This does not work with pool solo because such workers cannot communicate while working, only while idling.

    For pool solo, the number of workers returned is always the number of idle workers.
    """

    try:
        celery_ping = app.control.inspect().ping()
    except kombu.exceptions.OperationalError as e:
        logging.debug(f"Could not connect to Celery: {e}")
        return 0  # Without celery running there are no workers
    if celery_ping is None:
        return 0  # Celery is running and installed, but no workers are available
    return len(celery_ping)
