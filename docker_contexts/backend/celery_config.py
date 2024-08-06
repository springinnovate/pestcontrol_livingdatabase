from celery import Celery

def make_celery(app):
    # Instantiate Celery with the new settings
    celery = Celery(app.import_name, broker=app.config['broker_url'])
    # Update configuration to use the new keys
    celery.conf.update(
        broker_url=app.config['broker_url'],
        result_backend=app.config['result_backend']
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
